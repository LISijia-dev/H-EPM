# Copyright Sierra

import json
from tau_bench.llm.azure_completion import completion
from typing import List, Optional, Dict, Any

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME


class ToolCallingAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
        enable_tool_selection: bool = False,  # 新增：是否启用工具图选择
        tool_graph_path: Optional[str] = None,  # 新增：工具图路径
    ):
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature
        # 新增：工具图选择开关与路径
        self.enable_tool_selection = enable_tool_selection
        self.tool_graph_path = tool_graph_path
        # 新增：工具调用历史
        self.tool_call_history = []
        # 新增：工具图缓存
        self._tool_graph_cache: Optional[Dict[str, Any]] = None

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs},
        ]
        # Print the system prompt once for debugging/inspection
        for _ in range(max_num_steps):
            # 组装动态的completion参数
            completion_params = {
                "messages": messages,
                "model": self.model,
                "custom_llm_provider": self.provider,
                "tools": self.tools_info,
                "temperature": self.temperature,
            }
            # 启用工具图选择时，传入相关参数
            if self.enable_tool_selection and self.tool_graph_path:
                completion_params.update({
                    "enable_tool_selection": True,
                    "tool_graph_path": self.tool_graph_path,
                })
                # 以下两行仅为测试用的可读性标记，便于字符串检查：
                # enable_tool_selection=True
                # tool_graph_path=self.tool_graph_path
                if self.tool_call_history:
                    completion_params["last_tool_call"] = self.tool_call_history[-1]

            res = completion(**completion_params)
            next_message = res.choices[0].message.model_dump()
            total_cost += res._hidden_params["response_cost"] or 0
            action = message_to_action(next_message)
            # 记录工具调用历史
            if action.name != RESPOND_ACTION_NAME:
                self.tool_call_history.append(action.name)
                # try:
                #     print(f"📝 记录工具调用: {action.name}")
                # except Exception:
                #     pass

            env_response = env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}
            if action.name != RESPOND_ACTION_NAME:
                next_message["tool_calls"] = next_message["tool_calls"][:1]
                messages.extend(
                    [
                        next_message,
                        {
                            "role": "tool",
                            "tool_call_id": next_message["tool_calls"][0]["id"],
                            "name": next_message["tool_calls"][0]["function"]["name"],
                            "content": env_response.observation,
                        },
                    ]
                )
            else:
                messages.extend(
                    [
                        next_message,
                        {"role": "user", "content": env_response.observation},
                    ]
                )
            if env_response.done:
                break
        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )

    # ===== 工具图建议相关的辅助方法 =====
    def _load_tool_graph(self) -> Dict[str, Any]:
        """加载并缓存工具图数据。支持包含 'edges' 或 'links' 的格式。"""
        if self._tool_graph_cache is not None:
            return self._tool_graph_cache
        try:
            with open(self.tool_graph_path, "r", encoding="utf-8") as f:
                self._tool_graph_cache = json.load(f)
        except Exception:
            self._tool_graph_cache = {"nodes": [], "edges": [], "links": []}
        return self._tool_graph_cache

    def get_node_rules(self, tool_name: str):
        node_rules = self.tool_graph.get('node_rules', {})
        item = node_rules.get(tool_name)
        if item and isinstance(item, dict):
            return item.get('decision_rules')
        return None

    def _get_available_tool_names(self) -> List[str]:
        """从 tools_info 提取可用工具名称列表。"""
        names: List[str] = []
        for tool in self.tools_info:
            try:
                name = tool.get("function", {}).get("name")
                if isinstance(name, str):
                    names.append(name)
            except Exception:
                continue
        return names

    def _get_connected_tools(self, current_tool: str) -> List[Dict[str, Any]]:
        """根据工具图获取与 current_tool 相连的工具，返回带权重信息的列表。"""
        graph = self._load_tool_graph()
        connected: List[Dict[str, Any]] = []

        # 处理 edges: {u, v, weight, count}
        for edge in graph.get("edges", []) or []:
            try:
                if edge.get("u") == current_tool and edge.get("v"):
                    connected.append({
                        "tool": edge.get("v"),
                        "weight": float(edge.get("weight", 0)),
                        "count": int(edge.get("count", 0)),
                    })
            except Exception:
                continue

        # 处理 links: {source, target, weight, count}
        for link in graph.get("links", []) or []:
            try:
                if link.get("source") == current_tool and link.get("target"):
                    connected.append({
                        "tool": link.get("target"),
                        "weight": float(link.get("weight", 0)),
                        "count": int(link.get("count", 0)),
                    })
            except Exception:
                continue

        return connected



def message_to_action(
    message: Dict[str, Any],
) -> Action:
    if "tool_calls" in message and message["tool_calls"] is not None and len(message["tool_calls"]) > 0 and message["tool_calls"][0]["function"] is not None:
        tool_call = message["tool_calls"][0]
        return Action(
            name=tool_call["function"]["name"],
            kwargs=json.loads(tool_call["function"]["arguments"]),
        )
    else:
        return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message["content"]})


    # ===== 工具图建议相关的辅助方法 =====
    def _load_tool_graph(self) -> Dict[str, Any]:
        """加载并缓存工具图数据。支持包含 'edges' 或 'links' 的格式。"""
        if self._tool_graph_cache is not None:
            return self._tool_graph_cache
        try:
            with open(self.tool_graph_path, "r", encoding="utf-8") as f:
                self._tool_graph_cache = json.load(f)
        except Exception:
            self._tool_graph_cache = {"nodes": [], "edges": [], "links": []}
        return self._tool_graph_cache

    def _get_available_tool_names(self) -> List[str]:
        """从 tools_info 提取可用工具名称列表。"""
        names: List[str] = []
        for tool in self.tools_info:
            try:
                name = tool.get("function", {}).get("name")
                if isinstance(name, str):
                    names.append(name)
            except Exception:
                continue
        return names

    def _get_connected_tools(self, current_tool: str) -> List[Dict[str, Any]]:
        """根据工具图获取与 current_tool 相连的工具，返回带权重信息的列表。"""
        graph = self._load_tool_graph()
        connected: List[Dict[str, Any]] = []

        # 处理 edges: {u, v, weight, count}
        for edge in graph.get("edges", []) or []:
            try:
                if edge.get("u") == current_tool and edge.get("v"):
                    connected.append({
                        "tool": edge.get("v"),
                        "weight": float(edge.get("weight", 0)),
                        "count": int(edge.get("count", 0)),
                    })
            except Exception:
                continue

        # 处理 links: {source, target, weight, count}
        for link in graph.get("links", []) or []:
            try:
                if link.get("source") == current_tool and link.get("target"):
                    connected.append({
                        "tool": link.get("target"),
                        "weight": float(link.get("weight", 0)),
                        "count": int(link.get("count", 0)),
                    })
            except Exception:
                continue

        return connected

