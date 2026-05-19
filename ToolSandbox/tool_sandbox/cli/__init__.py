# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Run all scenarios in the tool sandbox."""

import argparse
import datetime
import json
import multiprocessing
import random
import subprocess
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Any, Optional, Union

import polars as pl
from tqdm import tqdm

from tool_sandbox.cli.utils import (
    AGENT_TYPE_TO_FACTORY,
    TEST_SCENARIO_NAMES,
    USER_TYPE_TO_FACTORY,
    RoleImplType,
    get_category_summary,
    get_category_to_scenario_count,
    get_necessary_tool_name_to_scenario_count,
    resolve_scenarios,
    run_scenario,
)
from tool_sandbox.common.execution_context import ScenarioCategories
from tool_sandbox.common.scenario import Scenario
from tool_sandbox.common.tool_discovery import ToolBackend

DEFAULT_USER_TYPE = RoleImplType.GPT_4_o_2024_05_13


def get_git_sha() -> Optional[str]:
    """Get the git SHA of the `HEAD` branch."""
    # From https://stackoverflow.com/a/21901260
    # Note that there are some 3rd party Python modules for interacting with git. I have
    # tried `pygit2` and `GitPython`, but both failed to get the commit associated with
    # `HEAD` for me.
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except subprocess.CalledProcessError:
        # The tool sandbox script was not executed from within the git repository so we
        # cannot figure out the git SHA.
        return None


def has_local_changes() -> bool:
    # From https://stackoverflow.com/a/3878934 . `git diff --exit-code` will return 0 if
    # there are no local changes. The `--quiet` suppresses printing to stdout. Note that
    # this approach does not detect untracked files, but this should be fine for our
    # purposes.
    completed_proc = subprocess.run(["git", "diff", "--exit-code", "--quiet"])
    return completed_proc.returncode == 1


def write_result_summary(
    result_summary: list[dict[str, Any]],
    category_summary: dict[str, dict[str, list[float]]],
    output_directory: Path,
    *,
    split: str,
    split_seed: int,
    split_ratio: float,
) -> None:
    # Try to get the current git SHA so that there is some provenance on with which
    # version of the code results have been generated with.
    git_sha = get_git_sha()
    if git_sha is not None and has_local_changes():
        git_sha += " + local changes"

    # Always create the directory chain immediately before writing (covers missing
    # mkdir in older forks, deleted run dirs, or resolve() edge cases on some FS).
    out_dir = Path(output_directory).expanduser()
    result_file = out_dir / "result_summary.json"
    try:
        result_file.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to create output directory '{result_file.parent}': {e}"
        ) from e
    if result_file.parent.exists() and not result_file.parent.is_dir():
        raise NotADirectoryError(
            f"Output path exists but is not a directory: '{result_file.parent}'"
        )

    with open(result_file, "w") as f:
        json.dump(
            {
                "per_scenario_results": result_summary,
                "category_aggregated_results": {
                    category: {k: sum(v) / len(v) for k, v in aggregation.items()}
                    for category, aggregation in category_summary.items()
                },
                "split": split,
                "split_seed": split_seed,
                "split_ratio": split_ratio,
                "git_sha": git_sha,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )


def run_sandbox(
    *,
    agent_type: RoleImplType,
    user_type: RoleImplType,
    name_to_scenario: dict[str, Scenario],
    processes: int,
    output_base_dir: Path,
    enable_summary: bool,
    split: str,
    split_seed: int,
    split_ratio: float,
    tool_graph_path: Optional[str] = None,
    enable_tool_suggestion: bool = False,
) -> None:
    """Entry point for Tool Sandbox

    Args:
        agent_type:       The agent type to use.
        user_type:        The user type to use.
        name_to_scenario: Dictionary from scenario name to scenario definition.
        processes:        Number of processes to run in parallel.
        output_base_dir:  Base directory for model outputs.

    """
    # Show all rows and all columns when converting polars dataframes to strings.
    # Sadly, there is no way to specify an unlimited format length for strings. Note
    # that for tracebacks or long explanations from Claude 3 Opus a value of `1000` was
    # insufficient.
    pl.Config.set_tbl_rows(-1).set_tbl_cols(-1).set_fmt_str_lengths(10000)
    pl.Config.set_tbl_formatting("ASCII_FULL")

    # Ensure output base directory exists
    output_base_dir = Path(output_base_dir).resolve()
    try:
        output_base_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to create output base directory '{output_base_dir}': {e}"
        ) from e

    # 创建 user（agent 将在 run_scenario 中创建）
    user = USER_TYPE_TO_FACTORY[user_type]()
    summary_suffix = "summary_on" if enable_summary else "summary_off"
    output_directory = (
        Path(output_base_dir)
        / f"H-EPM_{agent_type}_"
        f"user_{getattr(user, 'model_name', user_type)}_"
        f"{summary_suffix}_"
        f"split_{split}_"
        f"graph_{str(enable_tool_suggestion)}_"
        f"{datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"
    )
    print(f"Storing outputs to '{output_directory}'.")
    # Create run directory up front so result_summary.json can always be written even if
    # no scenario reaches run_scenario's mkdir (empty list, or failure before mkdir).
    output_directory.mkdir(parents=True, exist_ok=True)

    # Print a category-wise count before playing scenarios
    category_counter: Counter[Union[ScenarioCategories, str]] = (
        get_category_to_scenario_count(name_to_scenario)
    )
    print(
        "Number of test cases per category:",
        json.dumps(
            {str(k): v for k, v in category_counter.most_common(len(category_counter))},
            indent=4,
            ensure_ascii=False,
        ),
    )
    # Print a necessary tool-wise count before playing scenarios
    necessary_tool_counter: Counter[str] = get_necessary_tool_name_to_scenario_count(
        name_to_scenario
    )
    print(
        "Number of test cases per necessary tool name:",
        json.dumps(
            {
                str(k): v
                for k, v in necessary_tool_counter.most_common(
                    len(necessary_tool_counter)
                )
            },
            indent=4,
            ensure_ascii=False,
        ),
    )
    # Shuffle scenarios for load balancing
    name_and_scenario_list = list(name_to_scenario.items())
    random.shuffle(name_and_scenario_list)
    num_scenarios = len(name_and_scenario_list)
    
    # 禁用多进程执行，改为单线程执行以避免 "raise self._value" 错误
    print("⚠️  多进程已禁用，使用单线程执行以避免多进程错误")
    
    # 始终使用单线程执行
    result_summary = []
    tqdm_iterator = tqdm(name_and_scenario_list, desc="Scenarios")
    for name_and_scenario in tqdm_iterator:
        try:
            result = run_scenario(
                name_and_scenario,
                agent_type=agent_type,
                user_type=user_type,
                output_directory=output_directory,
                tool_graph_path=tool_graph_path,
                enable_tool_suggestion=enable_tool_suggestion,
                split=split,
            )
            result_summary.append(result)
        except Exception as e:
            print(f"❌ 场景执行失败: {name_and_scenario[0]} - {e}")
            # 添加一个失败的结果记录
            result_summary.append({
                "scenario_name": name_and_scenario[0],
                "error": str(e),
                "success": False
            })

    # Aggregate results by category
    category_summary = get_category_summary(result_summary)
    write_result_summary(
        result_summary=result_summary,
        category_summary=category_summary,
        output_directory=output_directory,
        split=split,
        split_seed=split_seed,
        split_ratio=split_ratio,
    )


def main() -> None:
    random.seed(42)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent",
        help="Agent type.",
        default="GPT_4_o_2024_05_13",
        choices=[str(t) for t in AGENT_TYPE_TO_FACTORY.keys()],
    )
    parser.add_argument(
        "--user",
        help="User type.",
        default=str(DEFAULT_USER_TYPE),
        choices=[str(t) for t in USER_TYPE_TO_FACTORY.keys()],
    )
    parser.add_argument(
        "--preferred_tool_backend",
        help="Preferred tool backend to use.",
        default="DEFAULT",
        choices=[str(t) for t in ToolBackend],
    )
    parser.add_argument(
        "--summary",
        help="Enable or disable summarize_the_task usage.",
        choices=["on", "off"],
        default="on",
    )
    parser.add_argument(
        "--split",
        help="Train/test split to run.",
        choices=["none", "train", "test"],
        default="none",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Seed for deterministic train/test split.",
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.8,
        help="Train ratio for deterministic split (0.0-1.0).",
    )
    scenario_selection_group = parser.add_mutually_exclusive_group()
    scenario_selection_group.add_argument(
        "-t",
        "--test_mode",
        action="store_true",
        help="Only run a few scenarios rather than the full suite.",
    )
    scenario_selection_group.add_argument(
        "-s",
        "--scenarios",
        nargs="*",
        help="Name of scenarios to run.",
        required=False,
    )
    parser.add_argument(
        "-p",
        "--parallel",
        type=int,
        default=16,
        help="Max number of processes for running scenarios in parallel.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=Path("data"),
        help="Output base directory.",
    )
    
    parser.add_argument(
        "--tool_graph_path",
        type=str,
        default=None,
        help="Path to tool graph JSON file for tool suggestion.",
    )

    parser.add_argument(
        "--enable_tool_suggestion",
        action="store_true",
        help="Enable tool suggestion based on tool graph.",

    )
    args = parser.parse_args()

    # The parser for `--test_mode` and `--scenarios` are in a mutually exclusive group
    # so we can safely ignore the value of `args.scenarios` when `args.test_mode` is
    # true.
    scenario_names = TEST_SCENARIO_NAMES if args.test_mode else args.scenarios

    name_to_scenario = resolve_scenarios(
        desired_scenario_names=scenario_names,
        preferred_tool_backend=args.preferred_tool_backend,
        enable_summary=(args.summary == "on"),
        split=args.split,
        split_seed=args.split_seed,
        split_ratio=args.split_ratio,
    )
    # Technically, strings can automatically be converted to the `RoleImplType` since it
    # is a `StrEnum`, but we are being explicit here.
    agent_type = RoleImplType(args.agent)
    user_type = RoleImplType(args.user)
    run_sandbox(
        agent_type=agent_type,
        user_type=user_type,
        name_to_scenario=name_to_scenario,
        processes=args.parallel,
        output_base_dir=args.output_dir,
        enable_summary=(args.summary == "on"),
        split=args.split,
        split_seed=args.split_seed,
        split_ratio=args.split_ratio,
        tool_graph_path=args.tool_graph_path,
        enable_tool_suggestion=args.enable_tool_suggestion,
    )


if __name__ == "__main__":
    main()
