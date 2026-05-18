#!/usr/bin/env python3
"""
Example usage of the updated Hugging Face model implementation
"""

import asyncio
from tau2.utils.llm_utils_local import HuggingFaceModel, generate
from tau2.data_model.message import UserMessage, SystemMessage

async def example_text_model():
    """Example using a text-only model"""
    print("=== Text Model Example ===")
    
    # Initialize a text model (e.g., Llama 2)
    model = HuggingFaceModel(
        model_name="meta-llama/Llama-2-7b-chat-hf",
        model_type="text",
        load_in_8bit=True,  # Use 8-bit quantization to save memory
    )
    
    # Build messages
    messages = model.build_messages(
        prompt="What is the capital of France?",
        system_prompt="You are a helpful assistant."
    )
    
    # Generate response
    response = await model.generate(
        messages,
        max_new_tokens=100,
        temperature=0.7,
    )
    
    print(f"Response: {response}")

async def example_vision_model():
    """Example using a vision-language model"""
    print("\n=== Vision Model Example ===")
    
    # Initialize a vision model
    model = HuggingFaceModel(
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        model_type="vision",
        load_in_4bit=True,  # Use 4-bit quantization for vision models
    )
    
    # Build messages with image
    messages = model.build_messages(
        prompt="What do you see in this image?",
        image_path="path/to/your/image.jpg",  # Replace with actual image path
        system_prompt="You are a helpful vision assistant."
    )
    
    # Generate response
    response = await model.generate(
        messages,
        max_new_tokens=200,
        temperature=0.5,
    )
    
    print(f"Response: {response}")

def example_using_generate_function():
    """Example using the generate function directly"""
    print("\n=== Generate Function Example ===")
    
    # Create messages
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="Explain quantum computing in simple terms.")
    ]
    
    # Generate response using the generate function
    response = generate(
        model_name="microsoft/DialoGPT-medium",
        messages=messages,
        model_type="text",
        max_new_tokens=150,
        temperature=0.8,
    )
    
    print(f"Response: {response.content}")

async def example_chat_conversation():
    """Example of a multi-turn conversation"""
    print("\n=== Chat Conversation Example ===")
    
    model = HuggingFaceModel(
        model_name="microsoft/DialoGPT-medium",
        model_type="text",
    )
    
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! How are you today?"},
    ]
    
    # First response
    response1 = await model.generate(conversation, max_new_tokens=50)
    print(f"Assistant: {response1}")
    
    # Add response to conversation
    conversation.extend([
        {"role": "assistant", "content": response1},
        {"role": "user", "content": "Can you help me with Python programming?"},
    ])
    
    # Second response
    response2 = await model.generate(conversation, max_new_tokens=100)
    print(f"Assistant: {response2}")

def main():
    """Run all examples"""
    print("Hugging Face Model Usage Examples")
    print("=" * 50)
    
    # Run async examples
    async def run_async_examples():
        await example_text_model()
        # Uncomment the line below if you have an image file
        # await example_vision_model()
        await example_chat_conversation()
    
    # Run async examples
    asyncio.run(run_async_examples())
    
    # Run sync example
    example_using_generate_function()

if __name__ == "__main__":
    main() 