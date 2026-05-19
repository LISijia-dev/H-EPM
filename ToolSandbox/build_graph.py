#!/usr/bin/env python3
"""
从指定trajectories文件夹中的所有conversation.json文件构建tool调用图。

输出格式:
{
  "nodes": [tool_name, ...],
  "edges": [
    {"u": src_tool, "v": dst_tool, "weight": float, "count": int, "current_information": []},
    ...
  ]
}

其中summarize_the_task不作为节点，而是将其调用信息作为边信息存储在前后两个tool之间。
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def extract_sequences_from_traj(traj: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    从轨迹中提取有序的tool调用名称/ID，并将summarize调用ID映射到其内容。

    返回:
        sequence: List of {"name": str, "id": Optional[str]}
        summarize_contents: Map from tool_call_id to summary content
    """
    sequence: List[Dict[str, Any]] = []
    summarize_contents: Dict[str, str] = {}

    # 第一遍：按顺序收集assistant的tool调用
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

    # 第二遍：收集summarize_the_task的tool回复
    for message in traj:
        if message.get("role") == "tool" and message.get("name") == "summarize_the_task":
            call_id = message.get("tool_call_id")
            content = message.get("content")
            if isinstance(call_id, str) and isinstance(content, str) and content.strip():
                summarize_contents[call_id] = content.strip()

    return sequence, summarize_contents


def load_result_summary(result_summary_path: Path) -> Dict[str, float]:
    """
    读取result_summary.json文件，返回scenario名称到milestone_similarity的映射。
    
    返回:
        Dict mapping scenario name to milestone_similarity
    """
    if not result_summary_path.exists():
        print(f"Warning: result_summary.json not found at {result_summary_path}")
        return {}
    
    try:
        with open(result_summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
        
        # result_summary.json 包含 per_scenario_results 数组
        per_scenario_results = summary_data.get("per_scenario_results", [])
        name_to_similarity = {}
        
        for scenario_result in per_scenario_results:
            name = scenario_result.get("name")
            milestone_similarity = scenario_result.get("milestone_similarity")
            if name is not None and milestone_similarity is not None:
                name_to_similarity[name] = float(milestone_similarity)
        
        print(f"Loaded {len(name_to_similarity)} scenarios from result_summary.json")
        return name_to_similarity
    except Exception as e:
        print(f"Error reading result_summary.json: {e}")
        return {}


def read_conversation_files(trajectories_dir: Path, name_to_similarity: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    读取trajectories文件夹中所有conversation.json文件，只保留milestone_similarity > 0.55的trajectory。
    
    参数:
        trajectories_dir: trajectories文件夹路径
        name_to_similarity: scenario名称到milestone_similarity的映射
    
    返回:
        List of conversation data, each containing {"traj": conversation_data, "file_path": str}
    """
    data = []
    conversation_files = list(trajectories_dir.rglob("conversation.json"))
    
    print(f"Found {len(conversation_files)} conversation.json files")
    
    filtered_count = 0
    for conv_file in conversation_files:
        try:
            # 从文件路径提取scenario名称
            # 假设路径格式为: scenario_name/conversation.json
            relative_path = conv_file.relative_to(trajectories_dir)
            scenario_name = relative_path.parent.name if relative_path.parent != Path(".") else relative_path.stem
            
            # 检查milestone_similarity
            milestone_similarity = name_to_similarity.get(scenario_name)
            if milestone_similarity is None:
                print(f"Skipping {relative_path}: scenario '{scenario_name}' not found in result_summary.json")
                continue
            
            if milestone_similarity <= 0.55:
                print(f"Skipping {relative_path}: milestone_similarity={milestone_similarity:.2f} <= 0.55")
                filtered_count += 1
                continue
            
            with open(conv_file, "r", encoding="utf-8") as f:
                conversation_data = json.load(f)
                if isinstance(conversation_data, list):
                    data.append({
                        "traj": conversation_data,
                        "file_path": str(relative_path)
                    })
                    print(f"Loaded: {relative_path} (milestone_similarity={milestone_similarity:.2f})")
                else:
                    print(f"Warning: {conv_file} does not contain a list")
        except Exception as e:
            print(f"Error reading {conv_file}: {e}")
    
    print(f"Filtered out {filtered_count} trajectories with milestone_similarity <= 0.55")
    print(f"Kept {len(data)} trajectories with milestone_similarity > 0.55")
    
    return data


def build_graph_from_data(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """从数据项列表构建tool调用图"""
    nodes_set = set()
    edge_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    # 累积逆轨迹长度的和作为边权重奖励
    edge_weight_bonuses: Dict[Tuple[str, str], float] = defaultdict(float)
    edge_infos: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for item in items:
        traj = item.get("traj") or item.get("messages") or []
        if not isinstance(traj, list):
            continue
        seq, summarize_contents = extract_sequences_from_traj(traj)

        # 更新节点（排除summarize_the_task本身，专注于可操作的tools）
        for entry in seq:
            name = entry.get("name")
            if name and name != "summarize_the_task":
                nodes_set.add(name)

        # 构建边
        seq_len = len(seq)
        inv_traj_len = 1.0 / seq_len if seq_len > 0 else 0.0
        
        # 1) 正常的连续边，跳过summarize节点
        for i in range(len(seq) - 1):
            u_name = seq[i].get("name")
            v_name = seq[i + 1].get("name")
            if not u_name or not v_name:
                continue
            if u_name == "summarize_the_task" or v_name == "summarize_the_task":
                continue
            edge_counts[(u_name, v_name)] += 1
            edge_weight_bonuses[(u_name, v_name)] += inv_traj_len

        # 2) 对于每个summarize出现，连接前后的非summary tools并附加summary内容
        for i, entry in enumerate(seq):
            if entry.get("name") != "summarize_the_task":
                continue
            # 找到前一个非summary tool
            prev_name = None
            for j in range(i - 1, -1, -1):
                n = seq[j].get("name")
                if n and n != "summarize_the_task":
                    prev_name = n
                    break
            # 找到下一个非summary tool
            next_name = None
            for k in range(i + 1, len(seq)):
                n = seq[k].get("name")
                if n and n != "summarize_the_task":
                    next_name = n
                    break
            if prev_name and next_name:
                edge_counts[(prev_name, next_name)] += 1
                edge_weight_bonuses[(prev_name, next_name)] += inv_traj_len
                # 如果可用，使用tool_call_id映射附加summary内容
                call_id = entry.get("id")
                if call_id and call_id in summarize_contents:
                    edge_infos[(prev_name, next_name)].append(summarize_contents[call_id])

    # 构建输出格式
    nodes = sorted(nodes_set)
    edges = []
    for (u, v), count in sorted(edge_counts.items()):
        edges.append(
            {
                "u": u,
                "v": v,
                # 权重结合原始计数和基于逆轨迹长度的奖励
                "weight": float(count) + float(edge_weight_bonuses.get((u, v), 0.0)),
                "count": count,
                "current_information": edge_infos.get((u, v), []),
            }
        )

    return {"nodes": nodes, "edges": edges}


def main():
    # 默认路径
    trajectories_dir = Path("/home/v-sijiali/project_s/tr-sandbox/data/suggqwen_Qwen_3_8B_user_gpt-4.1_summary_on_split_train_graph_False_01_03_2026_10_37_33")
    output_path = Path("tool_graph_qwen_8b_train.json")

    # CLI参数覆盖
    if len(sys.argv) > 1:
        trajectories_dir = Path(sys.argv[1]).resolve()
    if len(sys.argv) > 2:
        output_path = Path(sys.argv[2]).resolve()

    # 检查trajectories目录是否存在
    if not trajectories_dir.exists():
        print(f"Error: Trajectories directory does not exist: {trajectories_dir}")
        return

    # result_summary.json应该在trajectories_dir的父目录中
    result_summary_path = trajectories_dir.parent / "result_summary.json"
    if not result_summary_path.exists():
        # 如果父目录中没有，尝试在trajectories_dir本身中查找
        result_summary_path = trajectories_dir / "result_summary.json"
    
    print(f"Reading result_summary.json from: {result_summary_path}")
    name_to_similarity = load_result_summary(result_summary_path)
    
    if not name_to_similarity:
        print("Warning: No scenario similarity data loaded from result_summary.json.")
        print("All trajectories will be skipped (only trajectories with milestone_similarity > 0.55 are processed).")
    
    print(f"Reading conversation files from: {trajectories_dir}")
    
    # 读取所有conversation.json文件（已过滤，只保留milestone_similarity > 0.55的）
    data = read_conversation_files(trajectories_dir, name_to_similarity)
    
    if not data:
        print("No conversation data found!")
        return

    print(f"Processing {len(data)} conversation files...")
    
    # 构建图
    graph = build_graph_from_data(data)

    # 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    print(f"Tool graph written to: {output_path}")
    print(f"Graph contains {len(graph['nodes'])} nodes and {len(graph['edges'])} edges")
    
    # 打印一些统计信息
    if graph['nodes']:
        print(f"Nodes: {', '.join(graph['nodes'])}")
    
    if graph['edges']:
        print("Top edges by weight:")
        sorted_edges = sorted(graph['edges'], key=lambda x: x['weight'], reverse=True)
        for edge in sorted_edges[:10]:  # 显示前10个权重最高的边
            print(f"  {edge['u']} -> {edge['v']}: weight={edge['weight']:.2f}, count={edge['count']}")


if __name__ == "__main__":
    main()
