#!/usr/bin/env python3
"""
ToolACE-2-8B 模型使用示例
支持函数调用的 Llama 模型
"""

import asyncio
import json
from typing import List, Dict, Any
from tau2.utils.llm_utils_local import HuggingFaceModel
from tau2.data_model.message import UserMessage, SystemMessage, AssistantMessage

class ToolACEHandler:
    """ToolACE-2-8B 模型处理器，支持函数调用"""
    
    def __init__(self, model_name: str = "Team-ACE/ToolACE-2-8B", **kwargs):
        self.model = HuggingFaceModel(
            model_name=model_name,
            model_type="text",  # ToolACE 是基于 Llama 的文本模型
            load_in_8bit=True,  # 使用 8-bit 量化节省内存
            **kwargs
        )
        
    def format_tools_for_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """将工具定义格式化为 ToolACE 期望的格式"""
        tool_descriptions = []
        for tool in tools:
            name = tool.get("name", "")
            description = tool.get("description", "")
            parameters = tool.get("parameters", {})
            
            # 格式化参数
            param_str = ""
            if parameters:
                param_str = f"Parameters: {json.dumps(parameters, indent=2)}"
            
            tool_desc = f"Tool: {name}\nDescription: {description}\n{param_str}\n"
            tool_descriptions.append(tool_desc)
        
        return "\n".join(tool_descriptions)
    
    def build_function_call_prompt(
        self, 
        user_query: str, 
        tools: List[Dict[str, Any]], 
        system_prompt: str = None
    ) -> List[Dict[str, Any]]:
        """构建包含函数调用的提示"""
        
        if system_prompt is None:
            system_prompt = """You are a helpful assistant that can use tools to answer questions. 
When you need to use a tool, respond with the tool name and parameters in JSON format.
Available tools:"""

        # 格式化工具描述
        tools_description = self.format_tools_for_prompt(tools)
        
        # 构建完整提示
        full_prompt = f"{system_prompt}\n\n{tools_description}\n\nUser: {user_query}\nAssistant:"
        
        return [{"role": "user", "content": full_prompt}]
    
    async def generate_with_tools(
        self,
        user_query: str,
        tools: List[Dict[str, Any]],
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        **kwargs
    ) -> str:
        """使用工具生成响应"""
        
        messages = self.build_function_call_prompt(user_query, tools)
        
        response = await self.model.generate(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )
        
        return response
    
    async def generate_simple(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """简单文本生成"""
        
        messages = self.model.build_messages(prompt)
        
        response = await self.model.generate(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )
        
        return response

# 示例工具定义
EXAMPLE_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather information for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use"
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "calculate",
        "description": "Perform mathematical calculations",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    }
]

async def example_function_calling():
    """函数调用示例"""
    print("=== ToolACE-2-8B 函数调用示例 ===")
    
    handler = ToolACEHandler()
    
    # 示例查询
    queries = [
        "What's the weather like in Beijing?",
        "Calculate 15 * 23 + 7",
        "I need to know the temperature in Tokyo and then calculate 2^10"
    ]
    
    for query in queries:
        print(f"\n用户查询: {query}")
        
        try:
            response = await handler.generate_with_tools(
                user_query=query,
                tools=EXAMPLE_TOOLS,
                max_new_tokens=200,
                temperature=0.1
            )
            
            print(f"模型响应: {response}")
            
            # 尝试解析函数调用
            try:
                # 查找 JSON 格式的函数调用
                import re
                json_match = re.search(r'\{.*\}', response)
                if json_match:
                    function_call = json.loads(json_match.group())
                    print(f"解析的函数调用: {json.dumps(function_call, indent=2, ensure_ascii=False)}")
            except:
                print("未检测到函数调用格式")
                
        except Exception as e:
            print(f"生成失败: {e}")

async def example_simple_generation():
    """简单文本生成示例"""
    print("\n=== ToolACE-2-8B 简单文本生成示例 ===")
    
    handler = ToolACEHandler()
    
    prompts = [
        "Explain what is machine learning in simple terms.",
        "Write a short Python function to calculate factorial.",
        "What are the benefits of using open source software?"
    ]
    
    for prompt in prompts:
        print(f"\n提示: {prompt}")
        
        try:
            response = await handler.generate_simple(
                prompt=prompt,
                max_new_tokens=150,
                temperature=0.7
            )
            
            print(f"响应: {response}")
            
        except Exception as e:
            print(f"生成失败: {e}")

async def example_conversation():
    """对话示例"""
    print("\n=== ToolACE-2-8B 对话示例 ===")
    
    handler = ToolACEHandler()
    
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! Can you help me with programming?"},
    ]
    
    # 第一轮对话
    response1 = await handler.model.generate(conversation, max_new_tokens=100)
    print(f"助手: {response1}")
    
    # 添加响应到对话
    conversation.extend([
        {"role": "assistant", "content": response1},
        {"role": "user", "content": "How do I write a function in Python?"},
    ])
    
    # 第二轮对话
    response2 = await handler.model.generate(conversation, max_new_tokens=150)
    print(f"助手: {response2}")

def example_using_generate_function():
    """使用 generate 函数的示例"""
    print("\n=== 使用 generate 函数示例 ===")
    
    from tau2.utils.llm_utils_local import generate
    from tau2.data_model.message import UserMessage, SystemMessage
    
    messages = [
        SystemMessage(content="You are a helpful programming assistant."),
        UserMessage(content="Write a Python function to reverse a string.")
    ]
    
    try:
        response = generate(
            model_name="Team-ACE/ToolACE-2-8B",
            messages=messages,
            model_type="text",
            max_new_tokens=200,
            temperature=0.7,
            load_in_8bit=True
        )
        
        print(f"响应: {response.content}")
        
    except Exception as e:
        print(f"生成失败: {e}")

async def main():
    """运行所有示例"""
    print("ToolACE-2-8B 模型使用示例")
    print("=" * 50)
    
    # 运行函数调用示例
    await example_function_calling()
    
    # 运行简单生成示例
    await example_simple_generation()
    
    # 运行对话示例
    await example_conversation()
    
    # 运行 generate 函数示例
    example_using_generate_function()

if __name__ == "__main__":
    asyncio.run(main()) 