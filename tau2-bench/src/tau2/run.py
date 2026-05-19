import json
import multiprocessing
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from loguru import logger

from tau2.agent.llm_agent import LLMAgent, LLMGTAgent, LLMSoloAgent
from tau2.data_model.simulation import (AgentInfo, Info, Results, RunConfig,
                                        SimulationRun, UserInfo)
from tau2.data_model.message import AssistantMessage
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment, EnvironmentInfo
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.metrics.agent_metrics import compute_metrics
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import RegistryInfo, registry
from tau2.user.user_simulator import DummyUser, get_global_user_sim_guidelines
from tau2.utils.display import ConsoleDisplay, Text
from tau2.utils.pydantic_utils import get_pydantic_hash
from tau2.utils.tool_graph_utils import save_tool_graph, save_combined_tool_graphs, print_tool_graph, build_tool_graph_from_trajectory
from tau2.utils.utils import DATA_DIR, get_commit_hash, get_now, show_dict_diff


def get_options() -> RegistryInfo:
    """
    Returns options for the simulator.
    """
    return registry.get_info()


def get_environment_info(
    domain_name: str, include_tool_info: bool = False
) -> EnvironmentInfo:
    """Get information about the environment for a registered Domain"""
    global registry
    env_constructor = registry.get_env_constructor(domain_name)
    return env_constructor().get_info(include_tool_info=include_tool_info)


def load_tasks(task_set_name: str) -> list[Task]:
    """
    Loads the tasks for the given domain.
    """
    global registry
    task_loader = registry.get_tasks_loader(task_set_name)
    tasks = task_loader()
    return tasks


def get_tasks(
    task_set_name: str,
    task_ids: Optional[list[str]] = None,
    num_tasks: Optional[int] = None,
) -> list[Task]:
    """
    Loads the tasks for the given domain.
    """
    if task_ids is None:
        tasks = load_tasks(task_set_name=task_set_name)
    else:
        tasks = [
            task for task in load_tasks(task_set_name=task_set_name) if task.id in task_ids
        ]
    if task_ids is not None and len(tasks) != len(task_ids):
        missing_tasks = set(task_ids) - set([task.id for task in tasks])
        raise ValueError(
            f"Not all tasks were found for task set {task_set_name}: {missing_tasks}"
        )
    if num_tasks is not None:
        tasks = tasks[:num_tasks]
    return tasks


def make_run_name(config: RunConfig) -> str:
    """
    Make a run name from the run config
    """
    clean_llm_agent_name = config.llm_agent.split("/")[-1]
    agent_name = f"{config.agent}_{clean_llm_agent_name}"

    clean_llm_user_name = config.llm_user.split("/")[-1]
    user_name = f"{config.user}_{clean_llm_user_name}"

    summary_suffix = "_summary" if config.summary else ""
    
    return f"{get_now()}_{config.domain}_{agent_name}_{user_name}{summary_suffix}"


def run_domain(config: RunConfig) -> Results:
    """
    Run simulations for a domain
    """
    config.validate()
    ConsoleDisplay.display_run_config(config)
    if config.task_set_name is None:
        task_set_name = config.domain
    else:
        task_set_name = config.task_set_name
    tasks = get_tasks(task_set_name, config.task_ids, config.num_tasks)
    if "gt" in config.agent:
        total_num_tasks = len(tasks)
        tasks = [task for task in tasks if LLMGTAgent.check_valid_task(task)]
        num_tasks = len(tasks)
        console_text = Text(text=f"Running {num_tasks} out of {total_num_tasks} tasks for GT agent.", style="bold green")
        ConsoleDisplay.console.print(console_text)
    if "solo" in config.agent:
        total_num_tasks = len(tasks)
        tasks = [task for task in tasks if LLMSoloAgent.check_valid_task(task)]
        num_tasks = len(tasks)
        console_text = Text(text=f"Running {num_tasks} out of {total_num_tasks} tasks for solo agent.", style="bold green")
        ConsoleDisplay.console.print(console_text)



    ## 保存文件
    num_trials = config.num_trials
    save_to = config.save_to
    if save_to is None:
        # save_to = make_run_name(config)+"_2choosetool_summary_graph_noautosum_back920"
        save_to = 'H-EPM_' + make_run_name(config)
    save_to = DATA_DIR / "simulations" / f"{save_to}.json"
    simulation_results = run_tasks(
        domain=config.domain,
        tasks=tasks,
        agent=config.agent,
        user=config.user,
        llm_agent=config.llm_agent,
        llm_args_agent=config.llm_args_agent,
        llm_user=config.llm_user,
        llm_args_user=config.llm_args_user,
        num_trials=num_trials,
        max_steps=config.max_steps,
        max_errors=config.max_errors,
        save_to=save_to,
        console_display=True,
        evaluation_type=EvaluationType.ALL,
        max_concurrency=config.max_concurrency,
        seed=config.seed,
        log_level=config.log_level,
        summary=config.summary,
    )
    metrics = compute_metrics(simulation_results)
    ConsoleDisplay.display_agent_metrics(metrics)

    return simulation_results


def run_tasks(
    domain: str,
    tasks: list[Task],
    agent: str,
    user: str,
    llm_agent: Optional[str] = None,
    llm_args_agent: Optional[dict] = None,
    llm_user: Optional[str] = None,
    llm_args_user: Optional[dict] = None,
    num_trials: int = 1,
    max_steps: int = 100,
    max_errors: int = 10,
    save_to: Optional[str | Path] = None,
    console_display: bool = True,
    evaluation_type: EvaluationType = EvaluationType.ALL,
    max_concurrency: int = 1,
    seed: Optional[int] = 300,
    log_level: Optional[str] = "INFO",
    summary: bool = False,
    tool_graph_path: Optional[str] = None,
    enable_tool_selection: bool = False,
) -> Results:
    """
    Runs tasks for a given domain.
    If llm_as_judge is True, the LLM will be used to annotate the simulation run.
    Calculates the reward for the simulation run.
    Args:
        domain (str): The domain to run the simulation on.
        tasks (list[Task]): The tasks to run.
        agent (str): The agent to run the simulation on.
        user (str): The user to run the simulation on.
        llm_agent (str): The model to use for the agent.
        llm_args_agent (dict): The arguments to pass to the LLM for the agent.
        llm_user (str): The model to use for the user.
        llm_args_user (dict): The arguments to pass to the LLM for the user.
        max_steps (int): The maximum number of steps to run the simulation.
        max_errors (int): The maximum number of errors to allow in the simulation.
        save_to (str | Path): The path to json file where to save the simulation results. If the file already exists, it will try to resume the run.
        evaluation_type (EvaluationType): The type of evaluation to use.
        max_concurrency (int): The maximum number of concurrent simulations to run.
        seed (int): The seed to use for the simulation.
        log_level (str): The log level to use.
    Returns:
        The simulation results and the annotations (if llm_review is True).
    """
    if isinstance(save_to, str):
        save_to = Path(save_to)
    # Set log level from config
    logger.remove()
    logger.add(lambda msg: print(msg), level=log_level)
    if len(tasks) == 0:
        raise ValueError("No tasks to run")
    if num_trials <= 0:
        raise ValueError("Number of trials must be greater than 0")
    if max_steps <= 0:
        raise ValueError("Max steps must be greater than 0")
    if max_errors <= 0:
        raise ValueError("Max errors must be greater than 0")

    tool_graph_path = '/home/v-sijiali/project_s/apitau2-bench/tool_graph_gpt-4.1-mini.json'
    enable_tool_selection = True

    random.seed(seed)

    seeds = [random.randint(0, 1000000) for _ in range(num_trials)]
    if "seed" in llm_args_agent:
        logger.warning("Each trial will modify the seed for the agent")

    if "seed" in llm_args_user:
        logger.warning("Each trial will modify the seed for the user")

    lock = multiprocessing.Lock()

    info = get_info(
        domain=domain,
        agent=agent,
        user=user,
        llm_agent=llm_agent,
        llm_args_agent=llm_args_agent,
        llm_user=llm_user,
        llm_args_user=llm_args_user,
        num_trials=num_trials,
        max_steps=max_steps,
        max_errors=max_errors,
        seed=seed,
        summary=summary,
    )
    simulation_results = Results(
        info=info,
        tasks=tasks,
        simulations=[],
    )
    done_runs = set()
    if save_to is not None:
        # If save_to already exists, check if the user wants to resume the run.
        if save_to.exists():
            response = (
                ConsoleDisplay.console.input(
                    "[yellow]File [bold]{}[/bold] already exists. Do you want to resume the run? (y/n)[/yellow] ".format(
                        save_to
                    )
                )
                .lower()
                .strip()
            )
            if response != "y":
                raise FileExistsError(
                    f"File {save_to} already exists. Please delete it or use a different save_to name."
                )
            with open(save_to, "r") as fp:
                prev_simulation_results = Results.model_validate_json(fp.read())
                # Check if the run config has changed
                if get_pydantic_hash(prev_simulation_results.info) != get_pydantic_hash(
                    simulation_results.info
                ):
                    diff = show_dict_diff(
                        prev_simulation_results.info.model_dump(),
                        simulation_results.info.model_dump(),
                    )
                    ConsoleDisplay.console.print(
                        f"The run config has changed.\n\n{diff}\n\nDo you want to resume the run? (y/n)"
                    )
                    response = (
                        ConsoleDisplay.console.input(
                            "[yellow]File [bold]{}[/bold] already exists. Do you want to resume the run? (y/n)[/yellow] ".format(
                                save_to
                            )
                        )
                        .lower()
                        .strip()
                    )
                    if response != "y":
                        raise ValueError(
                            "The run config has changed. Please delete the existing file or use a different save_to name."
                        )
                # Check if the task set has changed
                if not all(
                    get_pydantic_hash(task) == get_pydantic_hash(prev_task)
                    for task, prev_task in zip(
                        sorted(simulation_results.tasks, key=lambda x: x.id),
                        sorted(prev_simulation_results.tasks, key=lambda x: x.id),
                    )
                ):
                    raise ValueError(
                        "The task set has changed. Please delete the existing file or use a different save_to name."
                    )
                # Check which of the runs have already been done
                done_runs = set(
                    [
                        (sim.trial, sim.task_id, sim.seed)
                        for sim in prev_simulation_results.simulations
                    ]
                )
                simulation_results = prev_simulation_results
                console_text = Text(text=f"Resuming run from {len(done_runs)} runs. {len(tasks) * num_trials - len(done_runs)} runs remaining.", style="bold yellow")
                ConsoleDisplay.console.print(console_text)
        # Create new save file
        else:
            # Check if save_to exists and create parent directories if needed
            if not save_to.parent.exists():
                save_to.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving simulation batch to {save_to}")
            with open(save_to, "w") as fp:
                fp.write(simulation_results.model_dump_json(indent=2))

    def _save(simulation: SimulationRun):
        if save_to is None:
            return
        with lock:
            with open(save_to, "r") as fp:
                ckpt = json.load(fp)
            
            # 直接使用simulation对象，它已经包含了任务信息
            simulation_data = simulation.model_dump()
            ckpt["simulations"].append(simulation_data)
            with open(save_to, "w") as fp:
                json.dump(ckpt, fp, indent=2)

    def _run(task: Task, trial: int, seed: int, progress_str: str) -> SimulationRun:
        console_text = Text(text=f"{progress_str}. Running task {task.id}, trial {trial + 1}", style="bold green")
        ConsoleDisplay.console.print(console_text)
        tool_graph_path = '/home/v-sijiali/project_s/apitau2-bench/tool_graph_gpt-4.1-mini.json'
        enable_tool_selection = True
        try:
            simulation = run_task(
                domain=domain,
                task=task,
                agent=agent,
                user=user,
                llm_agent=llm_agent,
                llm_args_agent=llm_args_agent,
                llm_user=llm_user,
                llm_args_user=llm_args_user,
                max_steps=max_steps,
                max_errors=max_errors,
                evaluation_type=evaluation_type,
                seed=seed,
                summary=summary,
                tool_graph_path=tool_graph_path,
                enable_tool_selection=enable_tool_selection,
            )
            simulation.trial = trial
            if console_display:
                ConsoleDisplay.display_simulation(simulation, show_details=False)
            
            # Print tool graph to console for debugging
            # if console_display:
            #     console_text = Text(text=f"Tool call sequence for task {task.id}, trial {trial + 1}:", style="bold yellow")
            #     ConsoleDisplay.console.print(console_text)
            #     print_tool_graph(simulation)
            
            _save(simulation)
        except Exception as e:
            logger.error(f"Error running task {task.id}, trial {trial}: {e}")
            raise e
        return simulation

    args = []
    for trial in range(num_trials):
        for i, task in enumerate(tasks):
            if (trial, task.id, seeds[trial]) in done_runs:
                console_text = Text(text=f"Skipping task {task.id}, trial {trial} because it has already been run.", style="bold yellow")
                ConsoleDisplay.console.print(console_text)
                continue
            progress_str = f"{i}/{len(tasks)} (trial {trial + 1}/{num_trials})"
            args.append((task, trial, seeds[trial], progress_str))

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        res = list(executor.map(_run, *zip(*args)))
        simulation_results.simulations.extend(res)
    ConsoleDisplay.console.print(
        "\n✨ [bold green]Successfully completed all simulations![/bold green]\n"
        "To review the simulations, run: [bold blue]tau2 view[/bold blue]"
    )
    
    # Write final aggregated results to disk
    if save_to is not None:
        try:
            with open(save_to, "w") as fp:
                fp.write(simulation_results.model_dump_json(indent=2))
        except Exception as e:
            logger.warning(f"Failed to write final simulation results to {save_to}: {e}")
    
    # Save combined tool graph for all trajectories
    # if save_to is not None and simulation_results.simulations:
    #     try:
    #         combined_tool_graph_path = save_to.with_name(f"{save_to.stem}_combined_tool_graphs.json")
    #         save_combined_tool_graphs(simulation_results.simulations, combined_tool_graph_path)
    #         console_text = Text(text=f"Combined tool graph saved to: {combined_tool_graph_path}", style="bold blue")
    #         ConsoleDisplay.console.print(console_text)
    #     except Exception as e:
    #         logger.warning(f"Failed to save combined tool graph: {e}")
        
    #     # Save individual tool graphs per trajectory (with reward embedded)
    #     try:
    #         per_traj_dir = save_to.with_name(f"{save_to.stem}_trajectory_graphs")
    #         per_traj_dir.mkdir(parents=True, exist_ok=True)
    #         for sim in simulation_results.simulations:
    #             # Prefer including task id, trial, and seed for uniqueness
    #             trial_str = getattr(sim, "trial", None)
    #             seed_str = getattr(sim, "seed", None)
    #             task_id = getattr(sim, "task_id", "unknown_task")
    #             parts = [str(task_id)]
    #             if trial_str is not None:
    #                 parts.append(f"trial{trial_str}")
    #             if seed_str is not None:
    #                 parts.append(f"seed{seed_str}")
    #             filename = "_".join(parts) + "_tool_graph.json"
    #             out_path = per_traj_dir / filename
    #             # Build graph and embed reward metadata
    #             graph = build_tool_graph_from_trajectory(sim)
    #             reward_value = None
    #             try:
    #                 reward_value = getattr(getattr(sim, "reward_info", None), "reward", None)
    #             except Exception:
    #                 reward_value = None
    #             payload = {
    #                 "task_id": task_id,
    #                 "trial": trial_str,
    #                 "seed": seed_str,
    #                 "reward": reward_value,
    #                 "nodes": graph.get("nodes", []),
    #                 "links": graph.get("links", []),
    #             }
    #             with open(out_path, "w", encoding="utf-8") as fp:
    #                 json.dump(payload, fp, ensure_ascii=False, indent=2)
    #         console_text = Text(text=f"Per-trajectory tool graphs saved under: {per_traj_dir}", style="bold blue")
    #         ConsoleDisplay.console.print(console_text)
    #     except Exception as e:
    #         logger.warning(f"Failed to save per-trajectory tool graphs: {e}")

    #     # Save final rewards per trajectory
    #     try:
    #         rewards = []
    #         for sim in simulation_results.simulations:
    #             reward_value = None
    #             try:
    #                 reward_value = getattr(getattr(sim, "reward_info", None), "reward", None)
    #             except Exception:
    #                 reward_value = None
    #             rewards.append({
    #                 "task_id": getattr(sim, "task_id", None),
    #                 "trial": getattr(sim, "trial", None),
    #                 "seed": getattr(sim, "seed", None),
    #                 "reward": reward_value,
    #             })
    #         rewards_path = save_to.with_name(f"{save_to.stem}_final_rewards.json")
    #         with open(rewards_path, "w", encoding="utf-8") as fp:
    #             json.dump(rewards, fp, ensure_ascii=False, indent=2)
    #         console_text = Text(text=f"Final rewards saved to: {rewards_path}", style="bold blue")
    #         ConsoleDisplay.console.print(console_text)
    #     except Exception as e:
    #         logger.warning(f"Failed to save final rewards: {e}")
    
    # Build and save a global tool graph aggregated across all simulations
    # try:
    #     _build_and_save_global_tool_graph(
    #         domain=domain,
    #         simulations=simulation_results.simulations,
    #         save_to=save_to,
    #     )
    # except Exception as e:
    #     logger.warning(f"Failed to build/save global tool graph: {e}")
        
    return simulation_results


def run_task(
    domain: str,
    task: Task,
    agent: str,
    user: str,
    llm_agent: Optional[str] = None,
    llm_args_agent: Optional[dict] = None,
    llm_user: Optional[str] = None,
    llm_args_user: Optional[dict] = None,
    max_steps: int = 100,
    max_errors: int = 10,
    evaluation_type: EvaluationType = EvaluationType.ALL,
    seed: Optional[int] = None,
    summary: bool = False,
    tool_graph_path: Optional[str] = '/home/v-sijiali/project_s/apitau2-bench/tool_graph_gpt-4.1-mini.json',
    enable_tool_selection: bool = True,
) -> SimulationRun:
    """
    Runs tasks for a given domain.
     If llm_as_judge is True, the LLM will be used to annotate the simulation run.
     Calculates the reward for the simulation run.
     Args:
         domain (str): The domain to run the simulation on.
         task (Task): The task to run.
         agent (str): The agent to run the simulation on.
         user (str): The user to run the simulation on.
         llm_agent (str): The model to use for the agent.
         llm_args_agent (dict): The arguments to pass to the LLM for the agent.
         llm_user (str): The model to use for the user.
         llm_args_user (dict): The arguments to pass to the LLM for the user.
         max_steps (int): The maximum number of steps to run the simulation.
         max_errors (int): The maximum number of errors to allow in the simulation.
         evaluation_type (EvaluationType): The type of evaluation to use.
         seed (int): The seed to use for the simulation.
     Returns:
         The simulation run.
    """

    if max_steps <= 0:
        raise ValueError("Max steps must be greater than 0")
    if max_errors <= 0:
        raise ValueError("Max errors must be greater than 0")
    global registry
    logger.info(
        f"STARTING SIMULATION: Domain: {domain}, Task: {task.id}, Agent: {agent}, User: {user}"
    )
    environment_constructor = registry.get_env_constructor(domain)
    environment = environment_constructor()
    AgentConstructor = registry.get_agent_constructor(agent)
    enable_tool_selection = True  # 强制开启
    tool_graph_path = '/home/v-sijiali/project_s/apitau2-bench/tool_graph_gpt-4.1-mini.json'

    solo_mode = False
    if issubclass(AgentConstructor, LLMAgent):
        # 检查是否启用工具选择功能
        if enable_tool_selection and tool_graph_path:
            logger.info(f"🎯 Enhanced tool selection enabled with graph: {tool_graph_path}")
            print(f"Using tool graph at: {tool_graph_path}")
            agent = AgentConstructor(
                tools=environment.get_tools(),
                domain_policy=environment.get_policy(),
                llm=llm_agent,
                llm_args=llm_args_agent,
                tool_graph_path=tool_graph_path,
                enable_tool_selection=True,
            )
            logger.info("✅ Created enhanced LLM agent with tool selection")
        else:
            agent = AgentConstructor(
                tools=environment.get_tools(),
                domain_policy=environment.get_policy(),
                llm=llm_agent,
                llm_args=llm_args_agent,
            )
            logger.info("✅ Created standard LLM agent")
    elif issubclass(AgentConstructor, LLMGTAgent):
        agent = AgentConstructor(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
            llm=llm_agent,
            llm_args=llm_args_agent,
            task=task,
        )
    elif issubclass(AgentConstructor, LLMSoloAgent):
        solo_mode = True
        environment: Environment = environment_constructor(solo_mode=True)
        user_tools = environment.get_user_tools() if environment.user_tools else []


        
        # 对于solo agent也可以使用增强版本
        if enable_tool_selection and tool_graph_path:
            logger.info(f"🎯 Enhanced tool selection enabled for solo agent with graph: {tool_graph_path}")
            print(f"Using tool graph at: {tool_graph_path}")
            agent = AgentConstructor(
                tools=environment.get_tools() + user_tools,
                domain_policy=environment.get_policy(),
                llm=llm_agent,
                llm_args=llm_args_agent,
                task=task,
                tool_graph_path=tool_graph_path,
                enable_tool_selection=True,
            )
            logger.info("✅ Created enhanced solo LLM agent with tool selection")
        else:
            agent = AgentConstructor(
                tools=environment.get_tools() + user_tools,
                domain_policy=environment.get_policy(),
                llm=llm_agent,
                llm_args=llm_args_agent,
                task=task,
            )
            logger.info("✅ Created standard solo LLM agent")
    else:
        raise ValueError(
            f"Unknown agent type: {AgentConstructor}. Should be LLMAgent or LLMSoloAgent"
        )
    
    # Add summary parameter to agent's llm_args
    if summary:
        agent.llm_args["summary"] = True

    try:
        user_tools = environment.get_user_tools()
    except Exception:
        user_tools = None

    UserConstructor = registry.get_user_constructor(user)
    if issubclass(UserConstructor, DummyUser):
        assert isinstance(agent, LLMSoloAgent), (
            "Dummy user can only be used with solo agent"
        )

    user = UserConstructor(
        tools=user_tools,
        instructions=str(task.user_scenario),
        llm=llm_user,
        llm_args=llm_args_user,
    )
    
    # Add summary parameter to user's llm_args
    if summary:
        user.llm_args["summary"] = True

    orchestrator = Orchestrator(
        domain=domain,
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        max_steps=max_steps,
        max_errors=max_errors,
        seed=seed,
        solo_mode=solo_mode,
    )
    simulation = orchestrator.run()

    reward_info = evaluate_simulation(
        domain=domain,
        task=task,
        simulation=simulation,
        evaluation_type=evaluation_type,
        solo_mode=solo_mode,
    )

    simulation.reward_info = reward_info

    logger.info(
        f"FINISHED SIMULATION: Domain: {domain}, Task: {task.id}, Agent: {agent.__class__.__name__}, User: {user.__class__.__name__}. Reward: {reward_info.reward}"
    )
    return simulation


def get_info(
    domain: str,
    agent: str,
    user: str,
    llm_agent: Optional[str] = None,
    llm_args_agent: Optional[dict] = None,
    llm_user: Optional[str] = None,
    llm_args_user: Optional[dict] = None,
    num_trials: int = 1,
    max_steps: int = 100,
    max_errors: int = 10,
    seed: Optional[int] = None,
    summary: bool = False,
) -> Info:
    user_info = UserInfo(
        implementation=user,
        llm=llm_user,
        llm_args=llm_args_user,
        global_simulation_guidelines=get_global_user_sim_guidelines(),
    )
    agent_info = AgentInfo(
        implementation=agent,
        llm=llm_agent,
        llm_args=llm_args_agent,
    )
    environment_info = get_environment_info(
        domain, include_tool_info=False
    )  # NOTE: Not saving tool info to avoid clutter.
    return Info(
        git_commit=get_commit_hash(),
        num_trials=num_trials,
        max_steps=max_steps,
        max_errors=max_errors,
        user_info=user_info,
        agent_info=agent_info,
        environment_info=environment_info,
        seed=seed,
        summary=summary,
    )


# def _build_and_save_global_tool_graph(domain: str, simulations: list[SimulationRun], save_to: Optional[Path]) -> None:
#     """
#     Build a global tool graph based on tool calls across all simulations.
#     - Add a node for each distinct tool used: {"id": name, "desc": "", "parameters": []}
#     - Add a directed link for consecutive tool calls: {"source": prev, "target": curr, "type": "semantic"}
#     - Print each node/link addition using JSON-like chunks following the reference format.
#     - Save the final graph JSON to disk next to the simulations file.
#     """
#     # Graph containers
#     nodes: list[dict] = []
#     links: list[dict] = []
#     seen_nodes: set[str] = set()
#     seen_links: set[tuple[str, str]] = set()

#     print(f"Building global tool graph for {domain} with {len(simulations)} simulations")

#     def _print_addition(kind: str, obj: dict) -> None:
#         if kind == "node":
#             print(json.dumps({"nodes": [obj]}, ensure_ascii=False, indent=2))
#         elif kind == "link":
#             print(json.dumps({"links": [obj]}, ensure_ascii=False, indent=2))

#     # Walk through all simulations in chronological tool call order per simulation
#     for sim in simulations:
#         # Flatten the tool call sequence across the conversation
#         ordered_tool_names: list[str] = []
#         for msg in sim.messages:
#             if isinstance(msg, AssistantMessage) and msg.tool_calls:
#                 for tc in msg.tool_calls:
#                     tool_name = tc.name
#                     # Add node if new
#                     if tool_name not in seen_nodes:
#                         node_obj = {"id": tool_name, "desc": "", "parameters": []}
#                         nodes.append(node_obj)
#                         seen_nodes.add(tool_name)
#                         _print_addition("node", node_obj)
#                     ordered_tool_names.append(tool_name)

#         # Build links between consecutive tool calls within this simulation
#         for i in range(1, len(ordered_tool_names)):
#             src = ordered_tool_names[i - 1]
#             tgt = ordered_tool_names[i]
#             edge = (src, tgt)
#             if edge not in seen_links:
#                 link_obj = {"source": src, "target": tgt, "type": "semantic"}
#                 links.append(link_obj)
#                 seen_links.add(edge)
#                 _print_addition("link", link_obj)

#     graph = {"nodes": nodes, "links": links}

#     # Decide output path
#     if isinstance(save_to, str):
#         save_path = Path(save_to)
#     else:
#         save_path = save_to or (DATA_DIR / "graph" / f"{get_now()}_{domain}_tool_graph.json")

#     if save_path.suffix != ".json":
#         # save_to might be a directory; ensure directory exists
#         (save_path).mkdir(parents=True, exist_ok=True)
#         out_path = save_path / f"{get_now()}_{domain}_tool_graph.json"
#     else:
#         # put graph next to the simulation json
#         out_path = save_path.with_name(save_path.stem + "_tool_graph.json")

#     # Ensure parent exists and write
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     with open(out_path, "w", encoding="utf-8") as fp:
#         json.dump(graph, fp, ensure_ascii=False, indent=2)
#     logger.info(f"Global tool graph saved to: {out_path}")
