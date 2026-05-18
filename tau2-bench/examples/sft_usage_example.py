#!/usr/bin/env python3
"""
Example of how to use SFT data collection in a real conversation
"""

import os
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tau2.agent.llm_agent import LLMAgent, LLMAgentState
from tau2.data_model.message import SystemMessage, UserMessage, AssistantMessage

def simulate_customer_service_conversation():
    """
    Simulate a customer service conversation and save it for SFT training
    """
    
    # Initialize agent state
    system_prompt = """You are a helpful customer service representative for a mobile phone company.
    Your goal is to help customers resolve their issues efficiently and professionally.
    Always be polite, patient, and try to understand the customer's problem before providing solutions."""
    
    state = LLMAgentState(
        system_messages=[SystemMessage(role="system", content=system_prompt)],
        messages=[]
    )
    
    # Simulate conversation turns
    conversation_turns = [
        ("user", "Hi, I'm having trouble with my mobile data. It's been very slow for the past few days."),
        ("assistant", "I'm sorry to hear you're experiencing slow mobile data. Let me help you troubleshoot this issue. Can you tell me when this problem started and if it's affecting all apps or just specific ones?"),
        ("user", "It started about 3 days ago and it's affecting everything - browsing, social media, even basic apps. I'm in the same area I usually am, so it's not a location issue."),
        ("assistant", "Thank you for the details. Since this is affecting all apps and started recently, let's try some basic troubleshooting steps. First, can you try turning your mobile data off and on again? Also, can you check if there are any pending system updates on your phone?"),
        ("user", "I already tried turning the data off and on, and my phone is up to date. The problem persists."),
        ("assistant", "I understand. Let me check your account for any potential issues. Can you confirm your phone number so I can look up your account details? Also, have you noticed if this happens at specific times of day or if it's constant throughout the day?"),
        ("user", "My number is 555-0123. The slow speed is constant throughout the day, not just at peak times."),
        ("assistant", "Thank you for confirming. I can see your account and there are no service restrictions or billing issues. Since this is a persistent problem affecting all apps, let me escalate this to our technical team. They'll investigate if there might be a network issue in your area or if there's a problem with your SIM card."),
        ("user", "That sounds good. How long will it take for them to investigate?"),
        ("assistant", "Our technical team typically responds within 24-48 hours. I'll also send you a text message with a reference number for this case. In the meantime, you can try using Wi-Fi when available, and if the problem gets worse, please call us back immediately. Is there anything else I can help you with today?")
    ]
    
    # Add conversation turns to state
    for turn_idx, (role, content) in enumerate(conversation_turns, 1):
        if role == "user":
            message = UserMessage(
                role="user",
                content=content,
                timestamp=datetime.now().isoformat(),
                turn_idx=turn_idx
            )
        else:
            message = AssistantMessage(
                role="assistant",
                content=content,
                timestamp=datetime.now().isoformat(),
                turn_idx=turn_idx
            )
        
        state.messages.append(message)
    
    return state

def main():
    """Main function demonstrating SFT data collection"""
    
    print("🎭 Simulating customer service conversation...")
    
    # Simulate the conversation
    conversation_state = simulate_customer_service_conversation()
    
    print(f"✅ Conversation simulation completed with {len(conversation_state.messages)} messages")
    
    # Create a mock agent (in real usage, this would be your actual LLM agent)
    class MockLLMAgent(LLMAgent):
        def __init__(self):
            self.llm = None
            self.tools = []
            self.llm_args = {}
            self.task = None
            self.domain_policy = ""
            self.provide_function_args = False
    
    agent = MockLLMAgent()
    
    # Save the conversation for SFT training
    print("\n💾 Saving conversation for SFT training...")
    
    try:
        # Save with a descriptive conversation ID
        filepath = agent.save_conversation_to_json(
            conversation_state, 
            "mobile_data_troubleshooting_001"
        )
        
        print(f"✅ Conversation saved successfully!")
        print(f"📁 File location: {filepath}")
        print(f"📊 Total messages saved: {len(conversation_state.messages)}")
        
        # Display conversation summary
        print(f"\n💬 Conversation Summary:")
        print(f"  - System Prompt: {conversation_state.system_messages[0].content[:80]}...")
        print(f"  - User Messages: {len([m for m in conversation_state.messages if m.role == 'user'])}")
        print(f"  - Assistant Messages: {len([m for m in conversation_state.messages if m.role == 'assistant'])}")
        print(f"  - Conversation ID: mobile_data_troubleshooting_001")
        
        print(f"\n🎯 This conversation data is now ready for SFT training!")
        
    except Exception as e:
        print(f"❌ Failed to save conversation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
