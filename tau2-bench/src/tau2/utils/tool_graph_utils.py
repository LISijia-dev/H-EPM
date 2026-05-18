"""
Utility functions for generating tool call graphs from simulation trajectories.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

from tau2.data_model.message import AssistantMessage, Message, ToolCall, UserMessage
from tau2.data_model.simulation import SimulationRun


def extract_tool_calls_from_messages(messages: List[Message]) -> List[ToolCall]:
    """
    Extract all tool calls from a list of messages in chronological order.
    
    Args:
        messages: List of messages from a simulation run
        
    Returns:
        List of tool calls in chronological order
    """
    tool_calls = []
    
    for message in messages:
        if isinstance(message, (AssistantMessage, UserMessage)) and message.tool_calls:
            tool_calls.extend(message.tool_calls)
    
    return tool_calls


def build_tool_graph_from_trajectory(simulation: SimulationRun) -> Dict:
    """
    Build a tool graph from a simulation trajectory.
    
    Args:
        simulation: The simulation run containing the trajectory
        
    Returns:
        Dictionary with 'nodes' and 'links' following sttool_graph_desc.json format
    """
    # Extract tool calls in chronological order
    tool_calls = extract_tool_calls_from_trajectory(simulation)
    
    # Build nodes and links
    nodes = []
    links = []
    seen_nodes: Set[str] = set()
    seen_links: Set[Tuple[str, str]] = set()
    
    # Create nodes for each unique tool
    for tool_call in tool_calls:
        tool_name = tool_call.name
        if tool_name not in seen_nodes:
            node = {
                "id": tool_name,
                "desc": "",  # Empty description as we don't have tool descriptions here
                "parameters": []  # Empty parameters as we don't have tool schema here
            }
            nodes.append(node)
            seen_nodes.add(tool_name)
    
    # Create links between consecutive tool calls
    for i in range(1, len(tool_calls)):
        prev_tool = tool_calls[i - 1].name
        curr_tool = tool_calls[i].name
        edge = (prev_tool, curr_tool)
        
        if edge not in seen_links:
            link = {
                "source": prev_tool,
                "target": curr_tool,
                "type": "semantic"
            }
            links.append(link)
            seen_links.add(edge)
    
    return {
        "nodes": nodes,
        "links": links
    }


def extract_tool_calls_from_trajectory(simulation: SimulationRun) -> List[ToolCall]:
    """
    Extract tool calls from a simulation run's messages in chronological order.
    
    Args:
        simulation: The simulation run
        
    Returns:
        List of tool calls in chronological order
    """
    return extract_tool_calls_from_messages(simulation.messages)


def save_tool_graph(simulation: SimulationRun, output_path: Path) -> None:
    """
    Generate and save a tool graph for a simulation trajectory.
    
    Args:
        simulation: The simulation run
        output_path: Path where to save the tool graph JSON file
    """
    tool_graph = build_tool_graph_from_trajectory(simulation)
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the tool graph
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tool_graph, f, ensure_ascii=False, indent=2)
    
    print(f"Tool graph saved to: {output_path}")


def print_tool_graph(simulation: SimulationRun) -> None:
    """
    Print the tool graph for a simulation trajectory in JSON format.
    
    Args:
        simulation: The simulation run
    """
    tool_graph = build_tool_graph_from_trajectory(simulation)
    
    # Print nodes
    if tool_graph["nodes"]:
        print(json.dumps({"nodes": tool_graph["nodes"]}, ensure_ascii=False, indent=2))
    
    # Print links
    if tool_graph["links"]:
        print(json.dumps({"links": tool_graph["links"]}, ensure_ascii=False, indent=2))


def get_tool_call_sequence(simulation: SimulationRun) -> List[str]:
    """
    Get the sequence of tool names called in a simulation.
    
    Args:
        simulation: The simulation run
        
    Returns:
        List of tool names in chronological order
    """
    tool_calls = extract_tool_calls_from_trajectory(simulation)
    return [tool_call.name for tool_call in tool_calls]


def _load_tool_desc_template() -> Dict[str, Dict]:
    """
    Load the domain tool description template and return a mapping of tool id -> node schema.

    Returns:
        Dict mapping tool id to node definition {"id","desc","parameters"}
    """
    # Default to telecom domain template location
    template_path = Path(__file__).parents[2] / "domains" / "telecom" / "tool_desc_graph_user.json"
    tool_id_to_node: Dict[str, Dict] = {}

    try:
        with open(template_path, "r", encoding="utf-8") as f:
            template = json.load(f)
        for node in template.get("nodes", []):
            node_id = node.get("id")
            if node_id:
                tool_id_to_node[node_id] = {
                    "id": node_id,
                    "desc": node.get("desc", ""),
                    "parameters": node.get("parameters", []),
                }
    except Exception:
        # If template missing or unreadable, fall back to empty mapping
        tool_id_to_node = {}

    return tool_id_to_node


def merge_tool_graphs_from_trajectories(simulations: List[SimulationRun]) -> Dict:
    """
    Merge tool graphs from multiple simulation trajectories into one large graph.
    
    Args:
        simulations: List of simulation runs
        
    Returns:
        Dictionary with merged 'nodes' and 'links' following tool_desc_graph_user.json format
    """
    # Load template node metadata (desc, parameters) if available
    tool_id_to_template_node = _load_tool_desc_template()

    # Gather unique tools that were actually invoked and collect ordered transitions per simulation
    called_tool_ids: Set[str] = set()
    seen_links: Set[Tuple[str, str]] = set()
    links: List[Dict] = []

    for simulation in simulations:
        tool_calls = extract_tool_calls_from_trajectory(simulation)
        for call in tool_calls:
            called_tool_ids.add(call.name)
        # Build links by call order within this simulation only
        for i in range(1, len(tool_calls)):
            prev_tool = tool_calls[i - 1].name
            curr_tool = tool_calls[i].name
            edge = (prev_tool, curr_tool)
            if edge not in seen_links:
                links.append({
                    "source": prev_tool,
                    "target": curr_tool,
                    "type": "semantic",
                })
                seen_links.add(edge)

    # Build nodes list using template metadata where possible; include only invoked tools
    nodes: List[Dict] = []
    for tool_id in sorted(called_tool_ids):
        if tool_id in tool_id_to_template_node:
            nodes.append(tool_id_to_template_node[tool_id])
        else:
            nodes.append({
                "id": tool_id,
                "desc": "",
                "parameters": [],
            })
    
    return {
        "nodes": nodes,
        "links": links
    }


def save_combined_tool_graphs(simulations: List[SimulationRun], output_path: Path) -> None:
    """
    Generate and save a merged tool graph from multiple simulation trajectories.
    Creates one large graph instead of separate graphs for each trajectory.
    
    Args:
        simulations: List of simulation runs
        output_path: Path where to save the merged tool graph JSON file
    """
    # Merge all trajectories into one large graph
    merged_graph = merge_tool_graphs_from_trajectories(simulations)
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the merged tool graph in tool_desc_graph_user.json format
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_graph, f, ensure_ascii=False, indent=2)
    
    print(f"Merged tool graph saved to: {output_path}")
    print(f"Total simulations: {len(simulations)}")
    print(f"Total unique tools: {len(merged_graph['nodes'])}")
    print(f"Total tool transitions: {len(merged_graph['links'])}")
