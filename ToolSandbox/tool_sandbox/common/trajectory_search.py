# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Utility to search for similar trajectories and extract tool call examples"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import polars as pl

from tool_sandbox.common.execution_context import ExecutionContext, DatabaseNamespace, RoleType


def load_trajectory(trajectory_dir: Path) -> Optional[ExecutionContext]:
    """Load execution context from a trajectory directory"""
    json_path = trajectory_dir / "execution_context.json"
    if not json_path.exists():
        return None
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return ExecutionContext.from_dict(data)


def extract_task_description(execution_context: ExecutionContext) -> str:
    """Extract the task description from execution context"""
    sandbox_db = execution_context.get_database(
        namespace=DatabaseNamespace.SANDBOX,
        get_all_history_snapshots=True,
        drop_sandbox_message_index=False,
    )
    
    # Find the first USER->AGENT message which should contain the task
    user_messages = sandbox_db.filter(
        (pl.col("sender") == "USER") & (pl.col("recipient") == "AGENT")
    )
    
    if len(user_messages) > 0:
        first_user_msg = user_messages.head(1).select("content")["content"][0]
        return first_user_msg
    
    return ""


def extract_tool_calls_from_trajectory(execution_context: ExecutionContext) -> List[Dict[str, Any]]:
    """Extract all tool calls from a trajectory"""
    sandbox_db = execution_context.get_database(
        namespace=DatabaseNamespace.SANDBOX,
        get_all_history_snapshots=True,
        drop_sandbox_message_index=False,
    )
    
    # Filter for AGENT->EXECUTION_ENVIRONMENT messages (tool calls)
    tool_call_messages = sandbox_db.filter(
        (pl.col("sender") == "AGENT") & (pl.col("recipient") == "EXECUTION_ENVIRONMENT")
    )
    
    tool_calls = []
    for row in tool_call_messages.to_dicts():
        if row.get("openai_function_name"):
            tool_calls.append({
                "function_name": row["openai_function_name"],
                "content": row.get("content", ""),
                "tool_call_id": row.get("openai_tool_call_id"),
            })
    
    return tool_calls


def simple_similarity(text1: str, text2: str) -> float:
    """Simple text similarity based on common words"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if len(words1) == 0 or len(words2) == 0:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if len(union) > 0 else 0.0


def find_most_similar_trajectory(
    trajectory_dir: Path,
    current_task: str
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Find the most similar trajectory and return its tool calls"""
    trajectories_dir = trajectory_dir / "trajectories"
    if not trajectories_dir.exists():
        return None, []
    
    best_similarity = -1
    best_trajectory_name = None
    best_tool_calls = []
    
    # Load all trajectories
    for traj_subdir in trajectories_dir.iterdir():
        if not traj_subdir.is_dir():
            continue
        
        execution_context = load_trajectory(traj_subdir)
        if execution_context is None:
            continue
        
        # Extract task description
        task_desc = extract_task_description(execution_context)
        
        # Calculate similarity
        similarity = simple_similarity(current_task, task_desc)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_trajectory_name = traj_subdir.name
            best_tool_calls = extract_tool_calls_from_trajectory(execution_context)
    
    return best_trajectory_name, best_tool_calls


def find_top_k_similar_trajectories(
    trajectory_dir: Path,
    current_task: str,
    k: int = 3,
) -> List[Tuple[str, List[Dict[str, Any]], float]]:
    """Return top-k similar trajectories with their tool calls and similarity.

    Returns a list of tuples: (trajectory_name, tool_calls, similarity), sorted by
    descending similarity and truncated to k entries.
    """
    trajectories_dir = trajectory_dir / "trajectories"
    if not trajectories_dir.exists():
        return []

    scored: List[Tuple[str, List[Dict[str, Any]], float]] = []

    for traj_subdir in trajectories_dir.iterdir():
        if not traj_subdir.is_dir():
            continue

        execution_context = load_trajectory(traj_subdir)
        if execution_context is None:
            continue

        task_desc = extract_task_description(execution_context)
        similarity = simple_similarity(current_task, task_desc)
        tool_calls = extract_tool_calls_from_trajectory(execution_context)
        scored.append((traj_subdir.name, tool_calls, similarity))

    # Sort by similarity desc, then by name for stability
    scored.sort(key=lambda x: (-x[2], x[0]))
    return scored[: max(0, k)]


def format_tool_calls_as_prompt(tool_calls: List[Dict[str, Any]]) -> str:
    """Format tool calls as a system prompt"""
    if not tool_calls:
        return ""
    
    prompt = "Based on similar historical tasks, here's a reference example of tool usage:\n\n"
    prompt += "**Example Tool Call Sequence:**\n\n"
    
    for i, tc in enumerate(tool_calls, 1):
        prompt += f"{i}. Called `{tc['function_name']}`\n"
        # Extract parameters from content if possible
        if tc.get('content'):
            # Try to extract function call parameters
            content = tc['content']
            prompt += f"   Content: {content[:200]}\n"
        prompt += "\n"
    
    return prompt


def get_similar_trajectory_prompt(data_dir: Path, current_task: str) -> Optional[str]:
    """
    Get a formatted prompt with similar trajectory tool calls
    
    Args:
        data_dir: Path to the data directory containing trajectories
        current_task: The current task description to match against
        
    Returns:
        Formatted prompt string or None if no similar trajectory found
    """
    traj_name, tool_calls = find_most_similar_trajectory(data_dir, current_task)
    
    if tool_calls:
        return format_tool_calls_as_prompt(tool_calls)
    
    return None


def format_top_k_trajectories_as_prompt(
    entries: List[Tuple[str, List[Dict[str, Any]], float]]
) -> str:
    """Format multiple trajectories as a single system prompt block."""
    if not entries:
        return ""

    prompt = (
        "Based on the most similar historical tasks, here are reference examples of tool usage (top 3):\n\n"
    )
    for rank, (name, tool_calls, similarity) in enumerate(entries, 1):
        prompt += f"### Example {rank} (similarity={similarity:.2f}, trajectory={name})\n\n"
        if not tool_calls:
            prompt += "(No tool calls recorded)\n\n"
            continue
        for i, tc in enumerate(tool_calls, 1):
            prompt += f"{i}. Called `{tc.get('function_name', 'unknown')}`\n"
            content = tc.get("content", "")
            if content:
                prompt += f"   Content: {content[:200]}\n"
        prompt += "\n"
    return prompt


def get_top_k_similar_trajectory_prompt(
    data_dir: Path, current_task: str, k: int = 3
) -> Optional[str]:
    """Get a formatted prompt for the top-k similar trajectories."""
    entries = find_top_k_similar_trajectories(data_dir, current_task, k=k)
    if not entries:
        return None
    return format_top_k_trajectories_as_prompt(entries)

