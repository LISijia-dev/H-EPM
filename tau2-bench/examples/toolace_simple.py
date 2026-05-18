#!/usr/bin/env python3
"""
ToolACE-2-8B 简化使用示例
"""

import asyncio
from tau2.utils.llm_utils_local import HuggingFaceModel, generate
from tau2.data_model.message import UserMessage, SystemMessage

async def simple_toolace_example():
    """最简单的 ToolACE-2-8B 使用示例"""
    
    print("=== ToolACE-2-8B 基础使用示例 ===")
    
    # 方法1: 使用 HuggingFaceModel 类
    model = HuggingFaceModel(
        model_name="Team-ACE/ToolACE-2-8B",
        model_type="text",
        load_in_8bit=True,  # 使用 8-bit 量化节省内存
    )
    
    # 构建消息
    messages = model.build_messages(
        prompt="What is the capital of France?",
        system_prompt="You are a helpful assistant."
    )
    
    # 生成响应
    response = await model.generate(
        messages,
        max_new_tokens=100,
        temperature=0.7,
    )
    
    print(f"方法1 - 响应: {response}")
    
    # 方法2: 使用 generate 函数
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="Explain what is Python programming language.")
    ]
    
    response = generate(
        model_name="Team-ACE/ToolACE-2-8B",
        messages=messages,
        model_type="text",
        max_new_tokens=150,
        temperature=0.7,
        load_in_8bit=True
    )
    
    print(f"方法2 - 响应: {response.content}")

async def function_calling_example():
    """函数调用示例"""
    
    print("\n=== ToolACE-2-8B 函数调用示例 ===")
    
    model = HuggingFaceModel(
        model_name="Team-ACE/ToolACE-2-8B",
        model_type="text",
        load_in_8bit=True,
    )
    
    # 定义工具
    tools_description = """
Available tools:
Tool: get_weather
Description: Get current weather information for a location
Parameters: {"type": "object", "properties": {"location": {"type": "string", "description": "The city name"}}, "required": ["location"]}

Tool: calculate
Description: Perform mathematical calculations
Parameters: {"type": "object", "properties": {"expression": {"type": "string", "description": "The mathematical expression"}}, "required": ["expression"]}
"""
    
    # 构建包含工具的提示
    prompt = f"""You are a helpful assistant that can use tools to answer questions.
When you need to use a tool, respond with the tool name and parameters in JSON format.

{tools_description}

User: What's the weather like in Beijing and calculate 15 * 23?
Assistant:"""
    
    messages = [{"role": "user", "content": prompt}]
    
    response = await model.generate(
        messages,
        max_new_tokens=200,
        temperature=0.1,  # 低温度以获得更确定的函数调用
    )
    
    print(f"函数调用响应: {response}")

async def conversation_example():
    """对话示例"""
    
    print("\n=== ToolACE-2-8B 对话示例 ===")
    
    model = HuggingFaceModel(
        model_name="Team-ACE/ToolACE-2-8B",
        model_type="text",
        load_in_8bit=True,
    )
    
    conversation = [
        {"role": "system", "content": "You are a helpful programming assistant."},
        {"role": "user", "content": "Hello! Can you help me learn Python?"},
    ]
    
    # 第一轮对话
    response1 = await model.generate(conversation, max_new_tokens=100)
    print(f"助手: {response1}")
    
    # 添加响应到对话
    conversation.extend([
        {"role": "assistant", "content": response1},
        {"role": "user", "content": "How do I write a function in Python?"},
    ])
    
    # 第二轮对话
    response2 = await model.generate(conversation, max_new_tokens=150)
    print(f"助手: {response2}")

def main():
    """运行所有示例"""
    print("ToolACE-2-8B 简化使用示例")
    print("=" * 50)
    
    async def run_examples():
        await simple_toolace_example()
        await function_calling_example()
        await conversation_example()
    
    asyncio.run(run_examples())

if __name__ == "__main__":
    main() 