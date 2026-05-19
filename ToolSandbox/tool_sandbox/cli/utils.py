# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import shutil
import traceback
import polars as pl
from collections import Counter, defaultdict
from enum import auto
from pathlib import Path
from typing import Any, Callable, Optional, Union

from strenum import StrEnum

from tool_sandbox.common.execution_context import (
    RoleType,
    ScenarioCategories,
    get_current_context,
    DatabaseNamespace,
)
from tool_sandbox.common.scenario import Scenario
from tool_sandbox.common.tool_discovery import ToolBackend
from tool_sandbox.roles.anthropic_api_agent import (
    ClaudeHaikuAgent,
    ClaudeOpusAgent,
    ClaudeSonnetAgent,
)
from tool_sandbox.roles.base_role import BaseRole
from tool_sandbox.roles.cli_role import CliAgent, CliUser
from tool_sandbox.roles.cohere_agent import CohereAgent
from tool_sandbox.roles.execution_environment import ExecutionEnvironment
from tool_sandbox.roles.gemini_agent import GeminiAgent
from tool_sandbox.roles.gorilla_api_agent import GorillaAPIAgent
from tool_sandbox.roles.hermes_api_agent import HermesAPIAgent
from tool_sandbox.roles.mistral_api_agent import MistralOpenAIServerAgent
from tool_sandbox.roles.openai_api_agent import (
    GPT_3_5_0125_Agent,
    GPT_4_0125_Agent,
    GPT_4_o_2024_05_13_Agent,
    GPT_4_1_Agent,
    GPT_4_1_mini_Agent,
    GPT_4o_Agent,
    GPT_5_Agent,
    Qwen_3_8B_Agent,
    Qwen_3_4B_Agent,
)
from tool_sandbox.roles.openai_api_user import (
    GPT_3_5_0125_User,
    GPT_4_0125_User,
    GPT_4_o_2024_05_13_User,
    GPT_4_1_User,
    GPT_4_1_mini_User,
    GPT_4o_User,
)
from tool_sandbox.roles.unhelpful_agent import UnhelpfulAgent
from tool_sandbox.scenarios import named_scenarios
from tool_sandbox.common.train_test_split import generate_and_save_splits


class RoleImplType(StrEnum):
    Hermes = auto()
    Gorilla = auto()
    GPT_3_5_0125 = auto()
    GPT_4_0125 = auto()
    GPT_4_o_2024_05_13 = auto()
    GPT_4_1 = auto()
    GPT_4_1_mini = auto()
    GPT_4_1_2024_04_14 = auto()
    GPT_4_1_2025_04_14 = auto()
    GPT_4o = auto()
    GPT_5 = auto()
    Qwen_3_8B = auto()
    Qwen_3_4B = auto()
    Claude_3_Opus = auto()
    Claude_3_Sonnet = auto()
    Claude_3_Haiku = auto()
    Gemini_1_0 = auto()
    Gemini_1_5 = auto()
    Gemini_1_5_Flash = auto()
    Cli = auto()
    Deterministic = auto()
    MistralOpenAIServer = auto()
    Cohere_Command_R = auto()
    Cohere_Command_R_Plus = auto()
    Unhelpful = auto()


AGENT_TYPE_TO_FACTORY: dict[RoleImplType, Callable[..., BaseRole]] = {
    RoleImplType.Hermes: lambda: HermesAPIAgent(
        model_name="NousResearch/Hermes-2-Pro-Mistral-7B"
    ),
    RoleImplType.Gorilla: lambda: GorillaAPIAgent(
        model_name="gorilla-llm/gorilla-openfunctions-v2"
    ),
    RoleImplType.MistralOpenAIServer: lambda: MistralOpenAIServerAgent(
        model_name="mistralai/Mistral-7B-Instruct-v0.3"
    ),
    RoleImplType.GPT_3_5_0125: GPT_3_5_0125_Agent,
    RoleImplType.GPT_4_0125: GPT_4_0125_Agent,
    RoleImplType.GPT_4_o_2024_05_13: GPT_4_o_2024_05_13_Agent,
    RoleImplType.GPT_4_1: lambda: GPT_4_1_Agent(),
    RoleImplType.GPT_4_1_mini: lambda: GPT_4_1_mini_Agent(),
    RoleImplType.GPT_4_1_2024_04_14: lambda: GPT_4_1_Agent(),
    RoleImplType.GPT_4_1_2025_04_14: lambda: GPT_4_1_Agent(),
    RoleImplType.GPT_4o: lambda: GPT_4o_Agent(),
    RoleImplType.GPT_5: lambda: GPT_5_Agent(),
    RoleImplType.Qwen_3_8B: lambda: Qwen_3_8B_Agent(),
    RoleImplType.Qwen_3_4B: lambda: Qwen_3_4B_Agent(),

    RoleImplType.Claude_3_Opus: ClaudeOpusAgent,
    RoleImplType.Claude_3_Sonnet: ClaudeSonnetAgent,
    RoleImplType.Claude_3_Haiku: ClaudeHaikuAgent,
    RoleImplType.Gemini_1_0: lambda: GeminiAgent(model_name="gemini-1.0-pro"),
    RoleImplType.Gemini_1_5: lambda: GeminiAgent(model_name="gemini-1.5-pro-001"),
    RoleImplType.Gemini_1_5_Flash: lambda: GeminiAgent(
        model_name="gemini-1.5-flash-001"
    ),
    RoleImplType.Cli: CliAgent,
    RoleImplType.Cohere_Command_R: lambda: CohereAgent(
        model_name="CohereForAI/c4ai-command-r-v01"
    ),
    RoleImplType.Cohere_Command_R_Plus: lambda: CohereAgent(
        model_name="CohereForAI/c4ai-command-r-plus"
    ),
    RoleImplType.Unhelpful: UnhelpfulAgent,
}

USER_TYPE_TO_FACTORY: dict[RoleImplType, Callable[..., BaseRole]] = {
    RoleImplType.GPT_3_5_0125: GPT_3_5_0125_User,
    RoleImplType.GPT_4_0125: GPT_4_0125_User,
    RoleImplType.GPT_4_o_2024_05_13: GPT_4_o_2024_05_13_User,
    RoleImplType.GPT_4_1: GPT_4_1_User,
    RoleImplType.GPT_4_1_mini: GPT_4_1_mini_User,
    RoleImplType.GPT_4_1_2024_04_14: GPT_4_1_User,
    RoleImplType.GPT_4_1_2025_04_14: GPT_4_1_User,
    RoleImplType.GPT_4o: GPT_4o_User,
    RoleImplType.Cli: CliUser,
}

# The scenarios to play back when the `--test_mode` flag is set.
TEST_SCENARIO_NAMES = [
    "send_message_with_contact_content_cellular_off_multiple_user_turn",
    "send_message_with_contact_content_cellular_off_multiple_user_turn_10_distraction_tools",
    "send_message_with_contact_content_cellular_off_3_distraction_tools_arg_description_scrambled",
    # "remove_contact_by_phone_multiple_user_turn",
    # "find_temperature_f_with_location_and_time_diff_multiple_user_turn",
]


def resolve_scenarios(
    desired_scenario_names: Optional[list[str]],
    preferred_tool_backend: ToolBackend,
    enable_summary: bool,
    *,
    split: str = "none",  # one of {"none", "train", "test"}
    split_seed: int = 42,
    split_ratio: float = 0.8,
) -> dict[str, Scenario]:
    """Resolve the scenarios to run.

    Args:
        desired_scenario_names: Name of scenarios to run. If empty all scenarios will be
                                returned.
        preferred_tool_backend: Which backend should be chosen in face of conflicting tool names.

    Returns:
        Dictionary from scenario name to definition.
    """
    if desired_scenario_names is None:
        # No filtering needed. Return all scenarios.
        scenarios = named_scenarios(preferred_tool_backend=preferred_tool_backend)
    else:
        scenarios = {
            name: scenario
            for name, scenario in named_scenarios(
                preferred_tool_backend=preferred_tool_backend
            ).items()
            if name in desired_scenario_names
        }

    # Optionally disable summarize tool and related system instruction
    if not enable_summary:
        for scenario in scenarios.values():
            # Remove summarize tool from allow list if present
            allow = scenario.starting_context.tool_allow_list
            if allow is not None and "summarize_the_task" in allow:
                scenario.starting_context.tool_allow_list = [
                    t for t in allow if t != "summarize_the_task"
                ]
            # Strip summarize requirement sentences from starting system -> agent message
            try:
                df = scenario.starting_context.get_database(
                    namespace=DatabaseNamespace.SANDBOX
                )
                if "content" in df.columns:
                    new_content = (
                        pl.when(pl.col("content").str.contains("summarize_the_task"))
                        .then(
                            pl.lit(
                                "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."
                            )
                        )
                        .otherwise(pl.col("content"))
                        .alias("content")
                    )
                    updated = df.with_columns(new_content)
                    scenario.starting_context.update_database(
                        namespace=DatabaseNamespace.SANDBOX, dataframe=updated
                    )
            except Exception:
                pass

    # If filtering by desired names, enforce subset again after modifications
    if desired_scenario_names is not None:
        name_to_scenario = {k: v for k, v in scenarios.items() if k in desired_scenario_names}
    else:
        name_to_scenario = scenarios

    # Deterministic train/test split if requested
    if split not in {"none", "train", "test"}:
        raise ValueError("split must be one of {'none', 'train', 'test'}")
    if split != "none":
        if not (0.0 < split_ratio < 1.0):
            raise ValueError("split_ratio must be in (0.0, 1.0)")
        # Also write split JSONs so they can be reused/inspected.
        # We generate per-module and combined splits under data/splits.
        try:
            generate_and_save_splits(
                preferred_tool_backend=preferred_tool_backend,
                split_seed=split_seed,
                split_ratio=split_ratio,
            )
        except Exception:
            # Non-fatal if writing fails; still perform in-memory split.
            pass
        # Use a deterministic shuffle based on provided seed
        scenario_names = sorted(name_to_scenario.keys())
        rng = __import__("random").Random(split_seed)
        rng.shuffle(scenario_names)
        split_index = int(len(scenario_names) * split_ratio)
        if split == "train":
            keep = set(scenario_names[:split_index])
        else:
            keep = set(scenario_names[split_index:])
        name_to_scenario = {k: v for k, v in name_to_scenario.items() if k in keep}

    # Raise an exception if not all desired scenarios exist, e.g. to fail if there was a
    # typo in the scenario names of the CLI command.
    if desired_scenario_names is not None:
        if len(desired_scenario_names) != len(name_to_scenario):
            missing_scenarios = set(desired_scenario_names) - set(name_to_scenario.keys())
            raise KeyError(
                "The following desired scenarios do not exist: "
                f"{sorted(list(missing_scenarios))}"
            )
    return name_to_scenario


def run_scenario(
    name_and_scenario: tuple[str, Scenario],
    *,
    agent_type: RoleImplType,
    user_type: RoleImplType,
    output_directory: Path,
    tool_graph_path: Optional[str] = None,
    enable_tool_suggestion: bool = False,
    split: str = "none",
) -> dict[str, Any]:
    """Play and evaluate a scenario.

    This is a necessary utility function to make multiprocessing work.

    Args:
        name_and_scenario:              Scenario name and Scenario object.
        agent_type:                     Agent type.
        user_type:                      User type.
        output_directory:               Directory to write output into.
        split:                          Split type ("train", "test", or "none").

    Returns:
        Evaluation info
    """
    name, scenario = name_and_scenario
    
    # 如果是train split，运行3次并选择最佳结果
    if split == "train":
        print(f"Running scenario '{name}' with train split...")
        return _run_scenario_multiple_times(
            name_and_scenario,
            agent_type=agent_type,
            user_type=user_type,
            output_directory=output_directory,
            tool_graph_path=tool_graph_path,
            enable_tool_suggestion=enable_tool_suggestion,
            num_runs=3,
        )
    else:
        # 非train split，运行一次
        return _run_scenario_single_time(
            name_and_scenario,
            agent_type=agent_type,
            user_type=user_type,
            output_directory=output_directory,
            tool_graph_path=tool_graph_path,
            enable_tool_suggestion=enable_tool_suggestion,
        )


def _run_scenario_single_time(
    name_and_scenario: tuple[str, Scenario],
    *,
    agent_type: RoleImplType,
    user_type: RoleImplType,
    output_directory: Path,
    tool_graph_path: Optional[str] = None,
    enable_tool_suggestion: bool = False,
) -> dict[str, Any]:
    """Run a scenario once and return the result."""
    name, scenario = name_and_scenario
    
    # 创建 agent，支持工具建议参数
    agent_factory = AGENT_TYPE_TO_FACTORY[agent_type]
    try:
        # 尝试无参数调用（兼容所有 agent）
        agent = agent_factory()
    except TypeError:
        # 如果无参数调用失败，尝试带参数调用（适用于支持工具建议的 agent）
        if agent_type in [RoleImplType.GPT_4_1, RoleImplType.GPT_4_1_2024_04_14, RoleImplType.GPT_4_1_2025_04_14, RoleImplType.GPT_4o,RoleImplType.GPT_5]:
            agent = agent_factory(tool_graph_path=tool_graph_path, enable_tool_suggestion=enable_tool_suggestion)
        else:
            # 对于其他不支持工具建议的 agent，使用无参数调用
            agent = agent_factory()
    
    roles = {
        RoleType.USER: USER_TYPE_TO_FACTORY[user_type](),
        RoleType.EXECUTION_ENVIRONMENT: ExecutionEnvironment(),
        RoleType.AGENT: agent,
    }
    output_directory.mkdir(parents=True, exist_ok=True)

    try:
        result = scenario.play_and_evaluate(
            roles=roles,
            output_directory=output_directory,
            scenario_name=name,
        )
        return {
            "name": name,
            "categories": scenario.categories,
            "traceback": None,
            "exception_type": None,
            "milestone_similarity": result.evaluation_result.milestone_similarity,
            "minefield_similarity": result.evaluation_result.minefield_similarity,
            "similarity": result.evaluation_result.similarity,
            "turn_count": result.evaluation_result.turn_count,
            "milestone_mapping": result.evaluation_result.milestone_mapping,
            "minefield_mapping": result.evaluation_result.minefield_mapping,
        }
    except Exception as e:
        return {
            "name": name,
            "categories": scenario.categories,
            "traceback": traceback.format_exc(),
            "exception_type": type(e).__name__,
            "milestone_similarity": 0,
            "minefield_similarity": 0,
            "similarity": 0,
            "turn_count": scenario.max_messages,
            "milestone_mapping": {},
            "minefield_mapping": {},
        }
    finally:
        for role in roles.values():
            role.teardown()


def _run_scenario_multiple_times(
    name_and_scenario: tuple[str, Scenario],
    *,
    agent_type: RoleImplType,
    user_type: RoleImplType,
    output_directory: Path,
    tool_graph_path: Optional[str] = None,
    enable_tool_suggestion: bool = False,
    num_runs: int = 3,
) -> dict[str, Any]:
    """Run a scenario multiple times and return the best result based on milestone_similarity and turn_count."""
    name, scenario = name_and_scenario
    best_result = None
    best_score = (-1, float('inf'))  # (milestone_similarity, turn_count)
    
    print(f"Running scenario '{name}' {num_runs} times for train split...")
    
    for run_idx in range(num_runs):
        # 创建临时目录用于这次运行
        temp_dir = output_directory / f"temp_run_{run_idx}"
        
        # 运行一次scenario
        result = _run_scenario_single_time(
            name_and_scenario,
            agent_type=agent_type,
            user_type=user_type,
            output_directory=temp_dir,
            tool_graph_path=tool_graph_path,
            enable_tool_suggestion=enable_tool_suggestion,
        )
        
        print(f"  Run {run_idx + 1}: milestone_similarity={result['milestone_similarity']:.3f}, turn_count={result['turn_count']}")
        
        # 检查是否是最佳结果：milestone_similarity最高，如果相同则选择turn_count最少
        current_score = (result['milestone_similarity'], result['turn_count'])
        if current_score > best_score:
            best_score = current_score
            best_result = result
            
            # 如果是更好的结果，将trajectory保存到主目录
            temp_trajectory_dir = temp_dir / "trajectories" / name
            main_trajectory_dir = output_directory / "trajectories" / name
            
            if temp_trajectory_dir.exists():
                # 删除之前的最佳结果
                if main_trajectory_dir.exists():
                    shutil.rmtree(main_trajectory_dir)
                
                # 复制当前最佳结果
                main_trajectory_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(temp_trajectory_dir, main_trajectory_dir)
        
        # 清理临时目录
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    print(f"  Best result: milestone_similarity={best_result['milestone_similarity']:.3f}, turn_count={best_result['turn_count']}")
    
    return best_result


def get_category_summary(
    result_summary: list[dict[str, Any]],
) -> dict[str, dict[str, list[float]]]:
    """Aggregate per test case result summary into category wise summary.

    Args:
        result_summary:     A list of results for each test case.

    Returns:
        Category wise summary.
    """
    # Aggregate results by category
    category_summary: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for current_summary in result_summary:
        for category in current_summary["categories"]:
            # The augmented scenarios are based on top of the `THREE_DISTRACTION_TOOLS`,
            # but we do not want to double count the stats for `THREE_DISTRACTION_TOOLS`.
            # Otherwise it would not be comparable to e.g. `TEN_DISTRACTION_TOOLS`.
            if category == ScenarioCategories.THREE_DISTRACTION_TOOLS and set(
                current_summary["categories"]
            ) & {
                ScenarioCategories.TOOL_NAME_SCRAMBLED,
                ScenarioCategories.TOOL_DESCRIPTION_SCRAMBLED,
                ScenarioCategories.ARG_DESCRIPTION_SCRAMBLED,
                ScenarioCategories.ARG_TYPE_SCRAMBLED,
                ScenarioCategories.ARG_NAME_SCRAMBLED,
            }:
                continue
            category_summary[category]["similarity"].append(
                current_summary["similarity"]
            )
            category_summary[category]["turn_count"].append(
                current_summary["turn_count"]
            )
        category_summary["ALL_CATEGORIES"]["similarity"].append(
            current_summary["similarity"]
        )
        category_summary["ALL_CATEGORIES"]["turn_count"].append(
            current_summary["turn_count"]
        )
    return category_summary


def get_category_to_scenario_count(
    name_to_scenario: dict[str, Scenario],
) -> Counter[Union[ScenarioCategories, str]]:
    """Count number of scenarios based on ScenarioCategories.

    Args:
        name_to_scenario:   A dict with scenario name as keys, scenario objects as values.

    Returns:
        A counter object containing counts for each category.
    """
    category_counter: Counter[Union[ScenarioCategories, str]] = Counter()
    for scenario in name_to_scenario.values():
        for category in scenario.categories:
            # The augmented scenarios are based on top of the `THREE_DISTRACTION_TOOLS`,
            # but we do not want to double count the stats for `THREE_DISTRACTION_TOOLS`.
            # Otherwise it would not be comparable to e.g. `TEN_DISTRACTION_TOOLS`.
            if category == ScenarioCategories.THREE_DISTRACTION_TOOLS and set(
                scenario.categories
            ) & {
                ScenarioCategories.TOOL_NAME_SCRAMBLED,
                ScenarioCategories.TOOL_DESCRIPTION_SCRAMBLED,
                ScenarioCategories.ARG_DESCRIPTION_SCRAMBLED,
                ScenarioCategories.ARG_TYPE_SCRAMBLED,
                ScenarioCategories.ARG_NAME_SCRAMBLED,
            }:
                continue
            category_counter[category] += 1
        category_counter["ALL_CATEGORIES"] += 1
    return category_counter


def get_necessary_tool_name_to_scenario_count(
    name_to_scenario: dict[str, Scenario],
) -> Counter[Union[ScenarioCategories, str]]:
    """Count number of scenarios based on necessary tool names.

    Args:
        name_to_scenario:   A dict with scenario name as keys, scenario objects as values.

    Returns:
        A counter object containing counts for each necessary tool names.
    """
    tool_name_counter: Counter[Union[ScenarioCategories, str]] = Counter(
        {
            tool_name: 0
            for tool_name in get_current_context().get_available_tools(
                scrambling_allowed=False
            )
        }
    )
    # Necessary tool names can be deducted from allowed tools in NO_DISTRACTION_TOOLS category
    # Then the total count equals the count from this category * number of augmentations.
    augmentation_categories: set[Union[ScenarioCategories, str]] = set()
    for scenario in name_to_scenario.values():
        if ScenarioCategories.NO_DISTRACTION_TOOLS in scenario.categories:
            assert scenario.starting_context.tool_allow_list is not None
            for necessary_tool in scenario.starting_context.tool_allow_list:
                tool_name_counter[necessary_tool] += 1
        augmentation_categories |= {
            ScenarioCategories.NO_DISTRACTION_TOOLS,
            ScenarioCategories.THREE_DISTRACTION_TOOLS,
            ScenarioCategories.TEN_DISTRACTION_TOOLS,
            ScenarioCategories.ALL_TOOLS_AVAILABLE,
            ScenarioCategories.TOOL_NAME_SCRAMBLED,
            ScenarioCategories.TOOL_DESCRIPTION_SCRAMBLED,
            ScenarioCategories.ARG_DESCRIPTION_SCRAMBLED,
            ScenarioCategories.ARG_TYPE_SCRAMBLED,
            ScenarioCategories.ARG_NAME_SCRAMBLED,
        } & set(scenario.categories)
    for necessary_tool in tool_name_counter:
        tool_name_counter[necessary_tool] *= len(augmentation_categories)
    return tool_name_counter
