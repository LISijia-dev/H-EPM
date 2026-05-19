# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Test case scenarios"""

import copy
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, cast

import polars as pl
from attrs import Factory, define
from tqdm import tqdm

from tool_sandbox.common.evaluation import (
    Evaluation,
    EvaluationResult,
    Milestone,
    MilestoneMatcher,
    Minefield,
)
from tool_sandbox.common.execution_context import (
    DatabaseNamespace,
    ExecutionContext,
    RoleType,
    ScenarioCategories,
    get_current_context,
    set_current_context,
)
from tool_sandbox.common.message_conversion import serialize_to_conversation
from tool_sandbox.roles.base_role import BaseRole

# 尝试导入更新图的函数（可选依赖）
try:
    # 优先尝试从同包导入
    from tool_sandbox.common.update_tool_graph import update_graph_from_conversation
    TOOL_GRAPH_UPDATE_AVAILABLE = True
    print("[TOOL_GRAPH_IMPORT] Successfully imported update_graph_from_conversation from tool_sandbox.common.update_tool_graph")
except (ImportError, AttributeError):
    # 如果同包导入失败，尝试其他方式
    try:
        import sys
        import importlib.util
        import os
        
        # 多种方式查找update_tool_graph.py
        # scenario.py位置: ToolSandbox-main/tool_sandbox/common/scenario.py
        # 所以 parent.parent.parent = ToolSandbox-main/
        base_path = Path(__file__).parent.parent.parent
        possible_paths = [
            # 方式1: 从scenario.py的位置向上找到项目根目录
            base_path / "update_tool_graph.py",
            # 方式2: 从当前工作目录查找
            Path.cwd() / "update_tool_graph.py",
            # 方式3: 从ToolSandbox-main目录的父目录查找（如果update_tool_graph.py在ToolSandbox-main同级）
            base_path.parent / "update_tool_graph.py",
            # 方式4: 从环境变量指定的路径查找
            Path(os.environ.get("TOOL_GRAPH_SCRIPT_PATH", "")) / "update_tool_graph.py" if os.environ.get("TOOL_GRAPH_SCRIPT_PATH") else None,
        ]
        # 过滤掉None值
        possible_paths = [p for p in possible_paths if p is not None]
        
        update_tool_graph_path = None
        for path in possible_paths:
            try:
                if path.exists() and path.is_file():
                    update_tool_graph_path = path.resolve()
                    print(f"[TOOL_GRAPH_IMPORT] Found update_tool_graph.py at: {update_tool_graph_path}")
                    break
            except Exception:
                continue
        
        if update_tool_graph_path:
            # 使用importlib直接加载模块
            spec = importlib.util.spec_from_file_location("update_tool_graph", update_tool_graph_path)
            if spec and spec.loader:
                update_tool_graph_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(update_tool_graph_module)
                update_graph_from_conversation = update_tool_graph_module.update_graph_from_conversation
                TOOL_GRAPH_UPDATE_AVAILABLE = True
                print("[TOOL_GRAPH_IMPORT] Successfully loaded update_tool_graph module")
            else:
                raise ImportError("Failed to create module spec")
        else:
            # 如果找不到文件，尝试常规导入（可能已经安装）
            print("[TOOL_GRAPH_IMPORT] update_tool_graph.py not found, trying standard import...")
            print(f"[TOOL_GRAPH_IMPORT] Searched paths: {[str(p) for p in possible_paths]}")
            from update_tool_graph import update_graph_from_conversation
            TOOL_GRAPH_UPDATE_AVAILABLE = True
            print("[TOOL_GRAPH_IMPORT] Successfully imported via standard import")
    except (ImportError, FileNotFoundError, AttributeError) as e:
        TOOL_GRAPH_UPDATE_AVAILABLE = False
        update_graph_from_conversation = None
        print(f"[TOOL_GRAPH_IMPORT] Failed to import update_tool_graph: {e}")
        print(f"[TOOL_GRAPH_IMPORT] This is not critical - tool graph updates will be skipped")


@define
class ScenarioResult:
    """Output of Scenario Play, saving both the execution context after the rollout is collected,
    and evaluation result

    """

    ending_context: ExecutionContext
    evaluation_result: EvaluationResult


@define
class Scenario:
    """Test case scenarios that defines a test case
    Each scenario contains an execution context defining starting state, and an evaluation object defining
    evaluation criteria
    """

    # Initial context, contains initial world state
    starting_context: ExecutionContext = Factory(ExecutionContext)
    # Evaluation definition
    evaluation: Evaluation = Factory(Evaluation)
    # Max number of total messages in roll out
    max_messages: int = 30
    # Category tags
    categories: List[ScenarioCategories] = Factory(list)

    def play(
        self, roles: Dict[RoleType, BaseRole], scenario_name: str
    ) -> ExecutionContext:
        """Play out the scenario and return execution context

        Args:
            roles:  A mapping indicating which Role we should use for each role type
            scenario_name: The scenario name.

        Returns:
            Execution context after playing out the scenario

        """
        execution_context = copy.deepcopy(self.starting_context)

        set_current_context(execution_context)
        # Prepare InteractiveConsole by consuming system message addressed to it
        sandbox_db = execution_context.get_database(
            DatabaseNamespace.SANDBOX,
            drop_sandbox_message_index=False,
            get_all_history_snapshots=True,
        )
        max_sandbox_message_index = execution_context.max_sandbox_message_index
        for message_index in range(max_sandbox_message_index + 1):
            if (
                sandbox_db["recipient"][message_index] == RoleType.EXECUTION_ENVIRONMENT
                and sandbox_db["sender"][message_index] == RoleType.SYSTEM
            ):
                roles[sandbox_db["recipient"][message_index]].respond(
                    ending_index=message_index
                )
        # Since this should only be processing system message, there should be no new messages after this
        assert (
            get_current_context().max_sandbox_message_index == max_sandbox_message_index
        )
        # Start processing non-system messages
        with tqdm(total=self.max_messages, desc=scenario_name) as pbar:
            while (
                sandbox_db["conversation_active"][-1]
                and sandbox_db["sandbox_message_index"][-1]
                < self.max_messages + max_sandbox_message_index
            ):
                roles[sandbox_db["recipient"][-1]].respond()
                sandbox_db = get_current_context().get_database(
                    DatabaseNamespace.SANDBOX, drop_sandbox_message_index=False
                )
                pbar.update(1)
            # Update max turns on successful end.
            pbar.total = pbar.n
            pbar.update(0)

        return get_current_context()

    def play_and_evaluate(
        self,
        roles: Dict[RoleType, BaseRole],
        output_directory: Path,
        scenario_name: str,
    ) -> ScenarioResult:
        """Play out the scenario and evaluate according to evaluation

        Args:
            roles:                      A mapping indicating which Role we should use for each role type
            output_directory:           Directory to write results to
            scenario_name:              Unique name for scenario. Used to serialize message history

        Returns:
            A ScenarioResult object containing the final execution context and evaluation result
            If play failed due to errors, return None object

        """
        # Prepare directories
        scenario_output_directory: Path = (
            output_directory / "trajectories" / scenario_name
        )
        scenario_output_directory.mkdir(exist_ok=True, parents=True)

        # If an exception occurs during playback we want to save the conversation and
        # execution context histories before re-raising the exception to skip
        # evaluation.
        try:
            self.play(roles=roles, scenario_name=scenario_name)
        except Exception:
            raise
        finally:
            execution_context = get_current_context()

            # Write pretty print messages
            # Skip user simulator few shot messages
            pretty_print_str = (
                "Note that User Simulator few shot messages have been omitted\n"
                + str(
                    execution_context.get_database(
                        DatabaseNamespace.SANDBOX,
                        get_all_history_snapshots=True,
                        drop_sandbox_message_index=False,
                    )
                    .filter(
                        (pl.col("visible_to") != [RoleType.USER])
                        | (pl.col("visible_to").is_null())
                    )
                    .drop(
                        [
                            "openai_tool_call_id",
                            "conversation_active",
                        ]
                    )
                )
            )
            with open(
                scenario_output_directory / "pretty_print.txt", "w", encoding="utf-8"
            ) as f:
                f.write(pretty_print_str)
            # Write execution_context
            with open(
                scenario_output_directory / "execution_context.json",
                "w",
                encoding="utf-8",
            ) as f:
                # We'll have to ditch dill InteractiveConsole here because
                # dill creates a bytes instead of raw string
                f.write(
                    json.dumps(
                        execution_context.to_dict(serialize_console=False),
                        ensure_ascii=False,
                        indent=4,
                    )
                )

        evaluation_result = self.evaluation.evaluate(
            execution_context=execution_context, max_turn_count=self.max_messages
        )

        # Write the conversation to a JSON file.
        conversation = serialize_to_conversation(
            execution_context=execution_context,
            evaluation_result=evaluation_result,
            milestones=cast(
                list[Milestone], self.evaluation.milestone_matcher.milestones
            ),
            minefields=cast(
                list[Minefield], self.evaluation.minefield_matcher.milestones
            ),
        )
        with open(
            scenario_output_directory / "conversation.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(conversation, f, indent=2, ensure_ascii=False)
        
        # 在每个trajectory完成后更新tool_graph.json（如果milestone_similarity > 0.7）
        if TOOL_GRAPH_UPDATE_AVAILABLE and update_graph_from_conversation:
            # 检查是否成功：milestone_similarity > 0.7 表示成功
            milestone_similarity = evaluation_result.milestone_similarity
            
            # 从环境变量或默认路径获取tool_graph.json路径
            tool_graph_path_str = os.environ.get(
                "TOOL_GRAPH_PATH", 
                str(output_directory.parent / "tool_graph.json")
            )
            tool_graph_path = Path(tool_graph_path_str)
            
            # 尝试更新图（函数内部会检查 milestone_similarity > 0.7）
            try:
                update_graph_from_conversation(
                    conversation=conversation,
                    graph_path=tool_graph_path,
                    backup_path=None,  # 自动生成备份
                    check_reward=False,  # 使用 milestone_similarity 判断，不检查 reward
                    milestone_similarity=milestone_similarity,
                )
            except Exception as e:
                # 不影响主流程，只打印错误
                print(f"[SCENARIO] Failed to update tool_graph.json: {e}")
        
        return ScenarioResult(
            ending_context=execution_context,
            evaluation_result=evaluation_result,
        )


@define
class ScenarioExtension:
    """Extends a few fields over base scenario to form a valid test scenario"""

    # Name for the resulting extended scenario
    name: str
    # Base scenario to extend on
    base_scenario: Scenario
    # Messages to extend to the starting context of base scenario
    messages: list[dict[str, Union[str, list[RoleType]]]] = Factory(list)
    # Tool allow list to extend to starting context of base scenario
    tool_allow_list: Optional[List[str]] = None
    # Tool deny list to extend to starting context of base scenario
    tool_deny_list: Optional[List[str]] = None
    # Evaluation milestones to extend to the evaluation of base scenario
    milestones: List[Milestone] = Factory(list)
    # Optional edge list defining Milestone dependencies. If None, creates a linked list
    milestone_edge_list: Optional[List[Tuple[int, int]]] = None
    # Evaluation minefields to extend to the evaluation of base scenario
    minefields: List[Minefield] = Factory(list)
    # Optional edge list defining Minefield dependencies. If None, creates a linked list
    minefield_edge_list: Optional[List[Tuple[int, int]]] = None
    # Categories to extend to scenario
    categories: List[ScenarioCategories] = Factory(list)

    def get_extended_scenario(self) -> Dict[str, Scenario]:
        """Get an extended scenario based on specified extensions

        Returns:
            A dictionary containing extended scenario and name
        """
        scenario: Scenario = copy.deepcopy(self.base_scenario)
        scenario.starting_context.add_to_database(
            namespace=DatabaseNamespace.SANDBOX, rows=self.messages
        )
        if self.tool_allow_list is not None:
            if scenario.starting_context.tool_allow_list is None:
                scenario.starting_context.tool_allow_list = []
            scenario.starting_context.tool_allow_list.extend(self.tool_allow_list)
        if self.tool_deny_list is not None:
            if scenario.starting_context.tool_deny_list is None:
                scenario.starting_context.tool_deny_list = []
            scenario.starting_context.tool_deny_list.extend(self.tool_deny_list)
        scenario.evaluation = Evaluation(
            milestone_matcher=MilestoneMatcher(
                milestones=self.milestones, edge_list=self.milestone_edge_list
            ),
            minefield_matcher=MilestoneMatcher(
                milestones=self.minefields, edge_list=self.minefield_edge_list
            ),
        )

        scenario.categories.extend(self.categories)
        return {self.name: scenario}
