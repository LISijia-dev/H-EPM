# Copyright Sierra

from typing import Any, Dict
from tau_bench.envs.tool import Tool


class SummarizeTheTask(Tool):
    @staticmethod
    # def invoke(data: Dict[str, Any], thought: str) -> str:
    #     return ""
    def invoke(data: Dict[str, Any], summary: str) -> str:
        return f"调用了 summarize_the_task，内容: {str(summary).strip()}"

    @staticmethod
    def get_info() -> Dict[str, Any]:
        """
        获取关于summarize函数的工具信息。
        
        返回:
            包含函数类型、名称、描述和参数规范的字典。
        """
        return {
            "type": "function",
            "function": {
                "name": "summarize_the_task",
                "description": "Please write a summary of the current state, including information from the environment and the user.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Please write a summary of the current state, including information from the environment and the user. A comprehensive summary of the current situation and the planned approach. Should include analysis of the current state, proposed plan, and comparison with any previous plans if applicable.",
                        },
                    },
                    "required": ["summary"],
                },
            },
        }
