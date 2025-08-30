#!/usr/bin/env python3

import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/home/cloud/Biomni')

def log_with_time(message):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def test_import():
    """Test importing A1 agent"""
    log_with_time("Testing A1 import...")
    try:
        from biomni.agent import A1
        log_with_time("✓ A1 import successful")
        return True
    except Exception as e:
        log_with_time(f"✗ A1 import failed: {e}")
        return False

def test_agent_creation_no_retriever():
    """Test creating A1 agent without tool retriever"""
    log_with_time("Testing A1 creation without tool retriever...")
    try:
        from biomni.agent import A1
        
        agent = A1(
            path='/mnt/exchange-saia/protein/haldhah/biomni_datalake',
            llm='custom',
            use_tool_retriever=False,  # Disable tool retriever
            timeout_seconds=30,
            base_url=os.environ.get("CUSTOM_MODEL_BASE_URL"),
            api_key=os.environ.get("CUSTOM_MODEL_API_KEY"),
            custom_model_name='qwen3-235b-a22b'
        )
        log_with_time("✓ A1 agent created successfully (no retriever)")
        return agent
    except Exception as e:
        log_with_time(f"✗ A1 agent creation failed (no retriever): {e}")
        import traceback
        traceback.print_exc()
        return None

def test_agent_creation_with_retriever():
    """Test creating A1 agent with tool retriever"""
    log_with_time("Testing A1 creation with tool retriever...")
    try:
        from biomni.agent import A1
        
        agent = A1(
            path='/mnt/exchange-saia/protein/haldhah/biomni_datalake',
            llm='custom',
            use_tool_retriever=True,  # Enable tool retriever
            timeout_seconds=30,
            base_url=os.environ.get("CUSTOM_MODEL_BASE_URL"),
            api_key=os.environ.get("CUSTOM_MODEL_API_KEY"),
            custom_model_name='qwen3-235b-a22b'
        )
        log_with_time("✓ A1 agent created successfully (with retriever)")
        return agent
    except Exception as e:
        log_with_time(f"✗ A1 agent creation failed (with retriever): {e}")
        import traceback
        traceback.print_exc()
        return None

def test_simple_query(agent):
    """Test simple query"""
    if agent is None:
        return False
        
    log_with_time("Testing simple query...")
    try:
        # Simple test query
        response = agent.go("What is 2+2?")
        log_with_time("✓ Simple query successful")
        log_with_time(f"Response preview: {str(response)[:100]}...")
        return True
    except Exception as e:
        log_with_time(f"✗ Simple query failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    log_with_time("=== BIOMNI A1 AGENT DEBUG SCRIPT ===")
    
    # Set environment variables
    os.environ["CUSTOM_MODEL_BASE_URL"] = "https://chat-ai.academiccloud.de/v1"
    os.environ["CUSTOM_MODEL_API_KEY"] = "YOUR_API_KEY_HERE"
    
    log_with_time(f"Environment: BASE_URL={os.environ.get('CUSTOM_MODEL_BASE_URL')}")
    log_with_time(f"Working directory: {os.getcwd()}")
    
    # Test 1: Import
    if not test_import():
        return
    
    # Test 2: Agent creation without retriever
    log_with_time("\n--- Testing without tool retriever ---")
    agent_no_retriever = test_agent_creation_no_retriever()
    
    if agent_no_retriever:
        # Test simple query
        test_simple_query(agent_no_retriever)
    
    # Test 3: Agent creation with retriever
    log_with_time("\n--- Testing with tool retriever ---")
    agent_with_retriever = test_agent_creation_with_retriever()
    
    if agent_with_retriever:
        # Test simple query
        test_simple_query(agent_with_retriever)
    
    log_with_time("\n=== DEBUG SCRIPT COMPLETED ===")

if __name__ == "__main__":
    main() 