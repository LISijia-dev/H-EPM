#!/usr/bin/env python3
"""
Extract a tool graph from dataexample.json with the same schema as tool_graph_data.json.

Output schema:
{
  "nodes": [tool_name, ...],
  "edges": [
    {"u": src_tool, "v": dst_tool, "weight": float, "count": int, "current_information": []},
    ...
  ]
}
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def extract_sequences_from_traj(traj: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Extract ordered tool call names/ids from a trajectory and map summarize call ids to their content.

    Returns:
        sequence: List of {"name": str, "id": Optional[str]}
        summarize_contents: Map from tool_call_id to summary content
    """
    sequence: List[Dict[str, Any]] = []
    summarize_contents: Dict[str, str] = {}

    # First pass: collect assistant tool calls in order
    for message in traj:
        if message.get("role") == "assistant":
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    call_id = None
                    name = None
                    if isinstance(tc, dict):
                        call_id = tc.get("id")
                        fn_obj = tc.get("function")
                        if isinstance(fn_obj, dict):
                            name = fn_obj.get("name")
                        if name is None:
                            name = tc.get("name")
                    if name:
                        sequence.append({"name": name, "id": call_id})

    # Second pass: collect tool replies for summarize_the_task
    for message in traj:
        if message.get("role") == "tool" and message.get("name") == "summarize_the_task":
            call_id = message.get("tool_call_id")
            content = message.get("content")
            if isinstance(call_id, str) and isinstance(content, str) and content.strip():
                summarize_contents[call_id] = content.strip()

    return sequence, summarize_contents


def build_graph_from_data(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    nodes_set = set()
    edge_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    # Accumulates the sum of inverse trajectory lengths as a bonus for edge weights
    edge_weight_bonuses: Dict[Tuple[str, str], float] = defaultdict(float)
    edge_infos: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for item in items:
        # Only process items with reward == 1.0
        reward = item.get("reward")
        if reward != 1.0:
            continue
        
        traj = item.get("traj") or item.get("messages") or []
        if not isinstance(traj, list):
            continue
        seq, summarize_contents = extract_sequences_from_traj(traj)

        # Update nodes (exclude summarize_the_task itself to focus on actionable tools)
        for entry in seq:
            name = entry.get("name")
            if name and name != "summarize_the_task":
                nodes_set.add(name)

        # Build edges
        seq_len = len(seq)
        inv_traj_len = 1.0 / seq_len if seq_len > 0 else 0.0
        # 1) Normal consecutive edges skipping summarize nodes
        for i in range(len(seq) - 1):
            u_name = seq[i].get("name")
            v_name = seq[i + 1].get("name")
            if not u_name or not v_name:
                continue
            if u_name == "summarize_the_task" or v_name == "summarize_the_task":
                continue
            edge_counts[(u_name, v_name)] += 1
            edge_weight_bonuses[(u_name, v_name)] += inv_traj_len

        # 2) For each summarize occurrence, connect previous and next non-summary tools and attach summary content
        for i, entry in enumerate(seq):
            if entry.get("name") != "summarize_the_task":
                continue
            # Find previous non-summary tool
            prev_name = None
            for j in range(i - 1, -1, -1):
                n = seq[j].get("name")
                if n and n != "summarize_the_task":
                    prev_name = n
                    break
            # Find next non-summary tool
            next_name = None
            for k in range(i + 1, len(seq)):
                n = seq[k].get("name")
                if n and n != "summarize_the_task":
                    next_name = n
                    break
            if prev_name and next_name:
                edge_counts[(prev_name, next_name)] += 1
                edge_weight_bonuses[(prev_name, next_name)] += 10*inv_traj_len
                # Attach summary content if available using tool_call_id mapping
                call_id = entry.get("id")
                if call_id and call_id in summarize_contents:
                    edge_infos[(prev_name, next_name)].append(summarize_contents[call_id])

    # Build output format
    nodes = sorted(nodes_set)
    edges = []
    for (u, v), count in sorted(edge_counts.items()):
        edges.append(
            {
                "u": u,
                "v": v,
                # Weight combines raw count with bonus based on inverse trajectory length
                "weight": float(count) + float(edge_weight_bonuses.get((u, v), 0.0)),
                "count": count,
                "current_information": edge_infos.get((u, v), []),
            }
        )

    return {"nodes": nodes, "edges": edges}


def main():
    # Paths
    root = Path(__file__).resolve().parent
    input_path = "/home/v-sijiali/project_s/gtau-bench/results/train-retail-train-tool-calling-gpt-4.1-0.0_range_0--1_user-gpt-4.1-llm_1114190946.json"  # 替换为实际文件
    output_path = root / "tool_graph_data_w10_gpt-4.1.json"

    # CLI overrides
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1]).resolve()
    if len(sys.argv) > 2:
        output_path = Path(sys.argv[2]).resolve()

    # Load input
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Expected top-level list in dataexample.json")

    graph = build_graph_from_data(data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    print(f"Wrote tool graph: {output_path}")


if __name__ == "__main__":
    main()


