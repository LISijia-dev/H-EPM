import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _extract_sequences_from_conversation(conversation: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    sequence: List[Dict[str, Any]] = []
    summarize_contents: Dict[str, str] = {}

    for message in conversation:
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

    for message in conversation:
        if message.get("role") == "tool" and message.get("name") == "summarize_the_task":
            call_id = message.get("tool_call_id")
            content = message.get("content")
            if isinstance(call_id, str) and isinstance(content, str) and content.strip():
                summarize_contents[call_id] = content.strip()

    return sequence, summarize_contents


def update_tool_graph_from_conversation(conversation_path: Path, graph_path: Path) -> None:
    if not conversation_path.exists():
        return

    try:
        with open(conversation_path, "r", encoding="utf-8") as f:
            conversation: List[Dict[str, Any]] = json.load(f)
    except Exception:
        return

    seq, summarize_contents = _extract_sequences_from_conversation(conversation)
    seq_len = len(seq)
    inv_traj_len = 1.0 / seq_len if seq_len > 0 else 0.0

    if graph_path.exists():
        try:
            with open(graph_path, "r", encoding="utf-8") as f:
                graph = json.load(f)
        except Exception:
            graph = {"nodes": [], "edges": [], "node_rules": {}}
    else:
        graph = {"nodes": [], "edges": [], "node_rules": {}}

    nodes: List[str] = list(graph.get("nodes", []))
    edges: List[Dict[str, Any]] = list(graph.get("edges", []))
    before_nodes = set(nodes)
    before_edges_count = len(edges)

    updated_edges: Dict[Tuple[str, str], Dict[str, Any]] = {}
    weight_changes: Dict[Tuple[str, str], Tuple[float, float]] = {}

    # Load canonical links from tool_des_converted.json to control allowed edges
    # Only update edges that exist in tool_des_converted.json; never create new edges
    def _load_allowed_links() -> tuple[set[tuple[str, str]], str]:
        candidates: List[Path] = []
        # 1) Same directory as the target graph
        candidates.append(graph_path.parent / "tool_graph_with_rules.json")
        # 2) Current working directory
        candidates.append(Path.cwd() / "tool_graph_with_rules.json")
        # 3) Environment override
        env_path = Path(json.loads(json.dumps(str(Path.cwd()))))  # no-op keep types happy
        env_file_str = os.environ.get("TOOL_DES_CONVERTED_PATH") if 'os' in globals() else None
        if env_file_str:
            candidates.append(Path(env_file_str))
        # 4) Project root guesses: traverse upwards from CWD to find pyproject.toml
        cur = Path.cwd()
        for _ in range(6):
            candidates.append(cur / "tool_graph_with_rules.json")
            if (cur / "pyproject.toml").exists() or (cur / ".git").exists():
                break
            if cur.parent == cur:
                break
            cur = cur.parent

        tried: List[str] = []
        for cand in candidates:
            tried.append(str(cand))
            try:
                if cand.exists():
                    with open(cand, "r", encoding="utf-8") as tf:
                        des_data = json.load(tf)
                    links = des_data.get("links", [])
                    allowed: set[tuple[str, str]] = set()
                    if isinstance(links, list):
                        for l in links:
                            s = l.get("source")
                            t = l.get("target")
                            if isinstance(s, str) and isinstance(t, str):
                                allowed.add((s, t))
                    if allowed:
                        return allowed, str(cand)
            except Exception:
                continue
        return set(), ";".join(tried)

    # Lazy import to avoid top import pollution
    import os  # noqa: E402
    allowed_links, tried_paths = _load_allowed_links()

    def find_edge(u: str, v: str) -> Dict[str, Any] | None:
        for e in edges:
            if e.get("u") == u and e.get("v") == v:
                return e
        return None

    def bump_edge(u: str, v: str, info: str | None) -> None:
        if not u or not v or u == "summarize_the_task" or v == "summarize_the_task":
            return
        # Respect canonical edge set from tool_des_converted.json
        if (u, v) not in allowed_links:
            return
        existing = find_edge(u, v)
        # Only update existing edges; do not create new edges/nodes
        if existing is None:
            return
        before_w = float(existing.get("weight", 0.0))
        existing["count"] = int(existing.get("count", 0)) + 1
        existing["weight"] = before_w + 1.0 + inv_traj_len
        after_w = float(existing.get("weight", before_w))
        if info:
            ci = existing.get("current_information") or []
            if info not in ci:
                ci.append(info)
            existing["current_information"] = ci
        updated_edges[(u, v)] = existing
        weight_changes[(u, v)] = (before_w, after_w)

    # Consecutive edges (skip summarize)
    for i in range(len(seq) - 1):
        u_name = seq[i].get("name")
        v_name = seq[i + 1].get("name")
        if not isinstance(u_name, str) or not isinstance(v_name, str):
            continue
        if u_name == "summarize_the_task" or v_name == "summarize_the_task":
            continue
        bump_edge(u_name, v_name, None)

    # Edges bridged by summarize_the_task with info
    summarize_attached: List[Tuple[str, str, str]] = []
    for i, entry in enumerate(seq):
        if entry.get("name") != "summarize_the_task":
            continue
        prev_name = None
        for j in range(i - 1, -1, -1):
            n = seq[j].get("name")
            if n and n != "summarize_the_task":
                prev_name = n
                break
        next_name = None
        for k in range(i + 1, len(seq)):
            n = seq[k].get("name")
            if n and n != "summarize_the_task":
                next_name = n
                break
        if prev_name and next_name:
            call_id = entry.get("id")
            info = summarize_contents.get(call_id) if isinstance(call_id, str) else None
            bump_edge(prev_name, next_name, info)
            if info:
                summarize_attached.append((prev_name, next_name, info))

    graph["nodes"] = sorted(nodes)
    graph["edges"] = edges

    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    print(f"[TOOL_GRAPH] Saved graph file: {graph_path}")

    # Print concise update summary
    added_nodes = sorted(set(nodes) - before_nodes)
    created_edges = max(0, len(edges) - before_edges_count)
    print(f"[TOOL_GRAPH] Updated graph: nodes={len(nodes)} (+{len(added_nodes)}), edges={len(edges)} (+{created_edges})")
    if updated_edges:
        print("[TOOL_GRAPH] Edges updated in this trajectory:")
        for (u, v), e in sorted(updated_edges.items()):
            count = e.get("count")
            weight = e.get("weight")
            info_count = len(e.get("current_information") or [])
            if (u, v) in weight_changes:
                bw, aw = weight_changes[(u, v)]
                print(f"  {u} -> {v}: count={count}, weight={bw:.3f} -> {aw:.3f}, info_items={info_count}")
            else:
                print(f"  {u} -> {v}: count={count}, weight={weight:.3f}, info_items={info_count}")
    else:
        if not allowed_links:
            print("[TOOL_GRAPH] No edges updated (allowed links empty). Tried: " + tried_paths)
        else:
            print("[TOOL_GRAPH] No edges updated (no matching existing and allowed edges in target graph)")
    if summarize_attached:
        print("[TOOL_GRAPH] Summarize info attached to edges:")
        for u, v, info in summarize_attached:
            preview = info.strip().replace("\n", " ")
            if len(preview) > 120:
                preview = preview[:117] + "..."
            print(f"  {u} -> {v}: summary=\"{preview}\"")


