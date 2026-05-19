import argparse
import json
import shutil
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


def is_success_traj(traj: List[Dict[str, Any]]) -> bool:
    """判断轨迹是否成功（reward == 1）。宽松扫描任意消息中的reward字段。"""
    for msg in traj or []:
        reward = msg.get("reward")
        if reward >0.7:
            return True
        # 兼容嵌套在tool结果或metadata内的场景
        for key in ("metadata", "result", "data"):
            container = msg.get(key)
            if isinstance(container, dict):
                r = container.get("reward")
                if r >0.7:
                    return True
    return False


def build_delta_from_traj(traj: List[Dict[str, Any]]) -> Tuple[set, Dict[Tuple[str, str], Dict[str, Any]]]:
    """
    基于一条轨迹构建需要合并到图中的增量：新增节点集合、边字典。
    边的结构：{"count": int, "weight": float, "infos": List[str]}
    """
    nodes_set = set()
    edge_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    edge_weight_bonuses: Dict[Tuple[str, str], float] = defaultdict(float)
    edge_infos: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    seq, summarize_contents = extract_sequences_from_traj(traj)

    # 更新节点（排除summarize_the_task）
    for entry in seq:
        name = entry.get("name")
        if name and name != "summarize_the_task":
            nodes_set.add(name)

    # 边构建
    seq_len = len(seq)
    inv_traj_len = 1.0 / seq_len if seq_len > 0 else 0.0

    # 连续边（跳过summary节点）
    for i in range(len(seq) - 1):
        u_name = seq[i].get("name")
        v_name = seq[i + 1].get("name")
        if not u_name or not v_name:
            continue
        if u_name == "summarize_the_task" or v_name == "summarize_the_task":
            continue
        edge_counts[(u_name, v_name)] += 1
        edge_weight_bonuses[(u_name, v_name)] += inv_traj_len

    # 将summary内容接到前后非summary工具之间
    for i, entry in enumerate(seq):
        if entry.get("name") != "summarize_the_task":
            continue
        # 找前一个非summary
        prev_name = None
        for j in range(i - 1, -1, -1):
            n = seq[j].get("name")
            if n and n != "summarize_the_task":
                prev_name = n
                break
        # 找后一个非summary
        next_name = None
        for k in range(i + 1, len(seq)):
            n = seq[k].get("name")
            if n and n != "summarize_the_task":
                next_name = n
                break
        if prev_name and next_name:
            edge_counts[(prev_name, next_name)] += 1
            edge_weight_bonuses[(prev_name, next_name)] += inv_traj_len
            call_id = entry.get("id")
            if call_id and call_id in summarize_contents:
                edge_infos[(prev_name, next_name)].append(summarize_contents[call_id])

    # 汇总成边增量
    delta_edges: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for key, count in edge_counts.items():
        delta_edges[key] = {
            "count": count,
            "weight": float(count) + float(edge_weight_bonuses.get(key, 0.0)),
            "infos": edge_infos.get(key, []),
        }
    return nodes_set, delta_edges


def merge_into_graph(graph: Dict[str, Any], add_nodes: set, add_edges: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Any]:
    """将增量节点与边合并进现有图结构。"""
    nodes = set(graph.get("nodes") or [])
    nodes.update(add_nodes)

    # 建立快速索引 (u,v) -> edge
    edges: List[Dict[str, Any]] = graph.get("edges") or []
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for e in edges:
        u = e.get("u")
        v = e.get("v")
        if isinstance(u, str) and isinstance(v, str):
            index[(u, v)] = e

    # 合并边
    for (u, v), delta in add_edges.items():
        if (u, v) in index:
            e = index[(u, v)]
            # 加权与计数累加
            try:
                e["weight"] = float(e.get("weight", 0.0)) + float(delta.get("weight", 0.0))
            except Exception:
                e["weight"] = float(delta.get("weight", 0.0))
            try:
                e["count"] = int(e.get("count", 0)) + int(delta.get("count", 0))
            except Exception:
                e["count"] = int(delta.get("count", 0))
            # 合并current_information
            infos = e.get("current_information")
            if not isinstance(infos, list):
                infos = []
            incoming_infos = delta.get("infos") or []
            if incoming_infos:
                infos.extend(incoming_infos)
            e["current_information"] = infos
        else:
            edges.append(
                {
                    "u": u,
                    "v": v,
                    "weight": float(delta.get("weight", 0.0)),
                    "count": int(delta.get("count", 0)),
                    "current_information": list(delta.get("infos") or []),
                }
            )

    # 排序保持稳定性
    graph["nodes"] = sorted(nodes)
    graph["edges"] = sorted(edges, key=lambda x: (x.get("u", ""), x.get("v", "")))
    return graph


def main():
    parser = argparse.ArgumentParser(description="Update tool_graph.json based on a single successful trajectory (reward==1)")
    parser.add_argument("trajectory", type=str, help="Path to a conversation/trajectory JSON file")
    parser.add_argument("tool_graph", type=str, nargs="?", default="tool_graph.json", help="Path to tool_graph.json (default: tool_graph.json)")
    parser.add_argument("backup_out", type=str, nargs="?", default=None, help="Optional backup output path for original graph")
    args = parser.parse_args()

    traj_path = Path(args.trajectory).resolve()
    graph_path = Path(args.tool_graph).resolve()

    if not traj_path.exists():
        print(f"Error: trajectory file not found: {traj_path}")
        return
    if not graph_path.exists():
        print(f"Error: tool_graph.json not found: {graph_path}")
        return

    # 读取轨迹数据
    with open(traj_path, "r", encoding="utf-8") as f:
        traj_data = json.load(f)
    # 支持外层为list或对象中含messages/traj
    if isinstance(traj_data, dict):
        traj = traj_data.get("traj") or traj_data.get("messages") or []
    else:
        traj = traj_data
    if not isinstance(traj, list):
        print("Error: trajectory format not supported (expected list of messages)")
        return

    # 仅成功轨迹才更新
    if not is_success_traj(traj):
        print("Trajectory reward!=1, skip updating tool_graph.json")
        return

    # 读取现有图并备份
    with open(graph_path, "r", encoding="utf-8") as f:
        original_graph = json.load(f)

    if args.backup_out:
        backup_path = Path(args.backup_out).resolve()
    else:
        backup_path = graph_path.with_name(graph_path.stem + "_backup.json")

    shutil.copyfile(graph_path, backup_path)
    print(f"Backed up original graph to: {backup_path}")

    # 生成增量并合并
    add_nodes, add_edges = build_delta_from_traj(traj)
    updated_graph = merge_into_graph(original_graph, add_nodes, add_edges)

    # 写回
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(updated_graph, f, ensure_ascii=False, indent=2)

    # 打印校验信息
    print(f"Updated tool graph written to: {graph_path}")
    print(f"Graph now contains {len(updated_graph.get('nodes', []))} nodes and {len(updated_graph.get('edges', []))} edges")

    # 列出此次增量涉及的边（按权重降序）
    if add_edges:
        print("Edges updated/added in this run:")
        # 构造展示行
        display_rows = []
        for (u, v), delta in add_edges.items():
            display_rows.append((u, v, float(delta.get("weight", 0.0)), int(delta.get("count", 0))))
        for u, v, w, c in sorted(display_rows, key=lambda x: x[2], reverse=True):
            print(f"  {u} -> {v}: +weight={w:.3f}, +count={c}")

    # 打印受summary影响的边，便于检查信息注入
    infos_total = 0
    for d in add_edges.values():
        infos_total += len(d.get("infos") or [])
    print(f"Appended {infos_total} summarize contents to edge current_information")


if __name__ == "__main__":
    main()


# ===== 可供内部调用的便捷方法 =====
def update_graph_from_messages(traj_messages: List[Dict[str, Any]], graph_path: str, backup_out: str | None = None, check_success: bool | None = None) -> bool:
    """从内存中的对话消息更新工具图。仅当成功时进行更新。

    参数:
        traj_messages: 对话消息列表
        graph_path: 工具图文件路径
        backup_out: 备份文件路径（可选）
        check_success: 如果提供，直接使用此值判断是否成功；否则使用 is_success_traj 从消息中判断

    返回 True 表示已更新（成功且写回成功），False 表示跳过或失败。
    """
    try:
        if not isinstance(traj_messages, list):
            print("[update_tool_graph] Provided messages is not a list, skip")
            return False

        # 如果提供了 check_success 参数，直接使用；否则从消息中判断
        if check_success is None:
            is_success = is_success_traj(traj_messages)
        else:
            is_success = check_success

        if not is_success:
            print("[update_tool_graph] Trajectory not successful, skip updating tool_graph.json")
            return False

        graph_p = Path(graph_path).resolve()
        if not graph_p.exists():
            print(f"[update_tool_graph] Graph file not found: {graph_p}")
            return False

        with open(graph_p, "r", encoding="utf-8") as f:
            original_graph = json.load(f)

        if backup_out:
            backup_path = Path(backup_out).resolve()
        else:
            backup_path = graph_p.with_name(graph_p.stem + "_backup.json")

        shutil.copyfile(graph_p, backup_path)
        print(f"[update_tool_graph] Backed up original graph to: {backup_path}")

        add_nodes, add_edges = build_delta_from_traj(traj_messages)
        updated_graph = merge_into_graph(original_graph, add_nodes, add_edges)

        with open(graph_p, "w", encoding="utf-8") as f:
            json.dump(updated_graph, f, ensure_ascii=False, indent=2)

        print(f"[update_tool_graph] Updated tool graph written to: {graph_p}")
        print(f"[update_tool_graph] Graph now contains {len(updated_graph.get('nodes', []))} nodes and {len(updated_graph.get('edges', []))} edges")

        if add_edges:
            print("[update_tool_graph] Edges updated/added in this run:")
            rows = []
            for (u, v), delta in add_edges.items():
                rows.append((u, v, float(delta.get("weight", 0.0)), int(delta.get("count", 0))))
            for u, v, w, c in sorted(rows, key=lambda x: x[2], reverse=True):
                print(f"  {u} -> {v}: +weight={w:.3f}, +count={c}")

        infos_total = sum(len(d.get("infos") or []) for d in add_edges.values())
        print(f"[update_tool_graph] Appended {infos_total} summarize contents to edge current_information")
        return True
    except Exception as e:
        print(f"[update_tool_graph] Failed to update graph from messages: {e}")
        return False


def update_graph_from_conversation(conversation: List[Dict[str, Any]], graph_path: str, backup_path: str | None = None, check_reward: bool = True, milestone_similarity: float | None = None) -> bool:
    """从对话中更新工具图。支持通过 milestone_similarity 判断成功。

    参数:
        conversation: 对话消息列表
        graph_path: 工具图文件路径
        backup_path: 备份文件路径（可选）
        check_reward: 是否检查 reward（默认 True，但会被 milestone_similarity 覆盖）
        milestone_similarity: milestone 相似度，如果提供且 > 0.7，则视为成功

    返回 True 表示已更新，False 表示跳过或失败。
    """
    # 优先使用 milestone_similarity 判断
    if milestone_similarity is not None:
        is_success = milestone_similarity > 0.7
        if not is_success:
            print(f"[update_tool_graph] milestone_similarity ({milestone_similarity}) <= 0.7, skip updating tool_graph.json")
            return False
    elif check_reward:
        # 如果没有提供 milestone_similarity，则使用原有的 reward 检查
        is_success = is_success_traj(conversation)
        if not is_success:
            print("[update_tool_graph] reward check failed, skip updating tool_graph.json")
            return False
    else:
        # 如果都不检查，直接更新
        is_success = True

    # 调用 update_graph_from_messages，传入 check_success 参数
    return update_graph_from_messages(
        traj_messages=conversation,
        graph_path=graph_path,
        backup_out=backup_path,
        check_success=is_success
    )


