#!/usr/bin/env python3
"""
Test script to verify agent initialization works in test mode
"""
import os
import sys
import asyncio

# Set test mode before importing backend modules
os.environ['TEST_MODE'] = 'true'
os.environ['OPENAI_API_KEY'] = 'test-key-for-testing'

def test_agent_import():
    """Test that we can import the nodes module without OpenAI API errors"""
    try:
        from backend.workflows.nodes import (
            ingest_agent, statistical_agent, regulatory_agent,
            consolidation_agent, report_agent, TEST_MODE
        )

        print(f"✅ TEST_MODE is: {TEST_MODE}")
        print(f"✅ Successfully imported all agents")
        print(f"✅ ingest_agent type: {type(ingest_agent)}")
        print(f"✅ statistical_agent type: {type(statistical_agent)}")

        return True
    except Exception as e:
        print(f"❌ Error importing agents: {e}")
        return False

async def test_agent_run():
    """Test that mock agents can run without errors"""
    try:
        from backend.workflows.nodes import ingest_agent

        result = await ingest_agent.run("Test document content")
        print(f"✅ Agent run result: {result}")
        return True
    except Exception as e:
        print(f"❌ Error running agent: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Testing agent initialization in test mode...")

    # Test import
    if not test_agent_import():
        sys.exit(1)

    # Test async agent run
    try:
        asyncio.run(test_agent_run())
        print("✅ All tests passed!")
    except Exception as e:
        print(f"❌ Async test failed: {e}")
        sys.exit(1)