#!/usr/bin/env python3
"""
Extract a tool graph from sft_data_reward_summary_solo_gt_1 folder with the same schema as tool_graph_data.json.
Only processes files where reward == 1.0

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
    
    # Summarize tool names to look for
    SUMMARIZE_TOOL_NAMES = {"summarize_the_task", "summarize_task_state"}

    # First pass: collect assistant tool calls in order
    for message in traj:
        if message.get("role") == "assistant":
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    call_id = None
                    name = None
                    arguments = None
                    if isinstance(tc, dict):
                        call_id = tc.get("id")
                        fn_obj = tc.get("function")
                        if isinstance(fn_obj, dict):
                            name = fn_obj.get("name")
                            arguments = fn_obj.get("arguments")
                        if name is None:
                            name = tc.get("name")
                        if arguments is None:
                            arguments = tc.get("arguments")
                    
                    if name:
                        sequence.append({"name": name, "id": call_id})
                        
                        # For summarize tools, extract content from arguments.current_information
                        if name in SUMMARIZE_TOOL_NAMES and call_id:
                            summary_content = None
                            if isinstance(arguments, dict):
                                summary_content = arguments.get("current_information")
                            elif isinstance(arguments, str):
                                # Arguments might be a JSON string
                                try:
                                    args_dict = json.loads(arguments)
                                    if isinstance(args_dict, dict):
                                        summary_content = args_dict.get("current_information")
                                except (json.JSONDecodeError, TypeError):
                                    pass
                            
                            if summary_content and isinstance(summary_content, str) and summary_content.strip():
                                summarize_contents[call_id] = summary_content.strip()

    return sequence, summarize_contents


def build_graph_from_data(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Summarize tool names to exclude from nodes and skip in edges
    SUMMARIZE_TOOL_NAMES = {"summarize_the_task", "summarize_task_state"}
    
    nodes_set = set()
    edge_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    # Accumulates the sum of inverse trajectory lengths as a bonus for edge weights
    edge_weight_bonuses: Dict[Tuple[str, str], float] = defaultdict(float)
    edge_infos: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for item in items:
        traj = item.get("traj") or item.get("messages") or []
        if not isinstance(traj, list):
            continue
        seq, summarize_contents = extract_sequences_from_traj(traj)

        # Update nodes (exclude summarize tools to focus on actionable tools)
        for entry in seq:
            name = entry.get("name")
            if name and name not in SUMMARIZE_TOOL_NAMES:
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
            if u_name in SUMMARIZE_TOOL_NAMES or v_name in SUMMARIZE_TOOL_NAMES:
                continue
            edge_counts[(u_name, v_name)] += 1
            edge_weight_bonuses[(u_name, v_name)] += inv_traj_len

        # 2) For each summarize occurrence, connect previous and next non-summary tools and attach summary content
        for i, entry in enumerate(seq):
            if entry.get("name") not in SUMMARIZE_TOOL_NAMES:
                continue
            # Find previous non-summary tool
            prev_name = None
            for j in range(i - 1, -1, -1):
                n = seq[j].get("name")
                if n and n not in SUMMARIZE_TOOL_NAMES:
                    prev_name = n
                    break
            # Find next non-summary tool
            next_name = None
            for k in range(i + 1, len(seq)):
                n = seq[k].get("name")
                if n and n not in SUMMARIZE_TOOL_NAMES:
                    next_name = n
                    break
            if prev_name and next_name:
                edge_counts[(prev_name, next_name)] += 1
                edge_weight_bonuses[(prev_name, next_name)] += inv_traj_len
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


def load_files_from_folder(folder_path: Path, require_reward: float = 1.0) -> List[Dict[str, Any]]:
    """
    Load all JSON files from a folder.
    Only include files where reward equals require_reward.
    
    Args:
        folder_path: Path to the folder containing JSON files
        require_reward: Only include files with this reward value (default: 1.0)
    
    Returns:
        List of valid data items
    """
    items = []
    json_files = list(folder_path.glob("*.json"))
    
    print(f"Found {len(json_files)} JSON files in {folder_path}")
    
    included_count = 0
    excluded_count = 0
    error_count = 0
    
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Check if reward matches
            reward = data.get("reward")
            if reward == require_reward:
                items.append(data)
                included_count += 1
            else:
                excluded_count += 1
                
        except Exception as e:
            error_count += 1
            print(f"Error reading {json_file.name}: {e}")
    
    print(f"Included: {included_count} files (reward == {require_reward})")
    print(f"Excluded: {excluded_count} files (reward != {require_reward})")
    if error_count > 0:
        print(f"Errors: {error_count} files")
    
    return items


def main():
    # Default paths
    root = Path(__file__).resolve().parent
    # input_folder = root / "data" / "messages" / "gpt-4.1-mini_sft_data_reward_summary"
    input_folder = Path("/home/v-sijiali/project_s/apitau2-bench/data/messages/sft_data_2")
    output_path = root / "tool_graph_gpt-4.1new2.json"

    # CLI overrides
    if len(sys.argv) > 1:
        input_folder = Path(sys.argv[1]).resolve()
    if len(sys.argv) > 2:
        output_path = Path(sys.argv[2]).resolve()

    if not input_folder.exists():
        print(f"Error: Input folder does not exist: {input_folder}")
        sys.exit(1)
    
    if not input_folder.is_dir():
        print(f"Error: Input path is not a directory: {input_folder}")
        sys.exit(1)

    # Load files with reward == 1.0
    items = load_files_from_folder(input_folder, require_reward=1.0)
    
    if not items:
        print("No valid files found with reward == 1.0")
        sys.exit(1)

    # Build graph
    graph = build_graph_from_data(items)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    print(f"\nWrote tool graph: {output_path}")
    print(f"  Nodes: {len(graph['nodes'])}")
    print(f"  Edges: {len(graph['edges'])}")


if __name__ == "__main__":
    main()
