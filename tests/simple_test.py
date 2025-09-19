#!/usr/bin/env python3
"""
Simple test script that bypasses complex imports
"""

import os
import sys
import pytest
import asyncio

# Set test mode
os.environ['TEST_MODE'] = 'true'
os.environ['GROQ_API_KEY'] = 'test-key'

def test_basic_imports():
    """Test basic module imports"""
    from backend.core.config import settings
    from backend.database.duckdb_manager import DuckDBManager
    from backend.services.document_processor import DocumentProcessor
    from backend.services.statistical_analyzer import StatisticalAnalyzer
    assert settings is not None
    print("✅ Basic imports successful")

def test_agent_mock_imports():
    """Test agent imports in test mode"""
    from backend.workflows.nodes import ingest_agent, TEST_MODE
    assert TEST_MODE == True
    assert hasattr(ingest_agent, 'run')
    print("✅ Agent mock imports successful")

@pytest.mark.asyncio
async def test_mock_agent_run():
    """Test that mock agents work"""
    from backend.workflows.nodes import ingest_agent
    result = await ingest_agent.run("test content")
    assert result is not None
    assert 'status' in result
    print(f"✅ Mock agent test: {result}")

def test_statistical_analyzer():
    """Test statistical analyzer"""
    from backend.services.statistical_analyzer import StatisticalAnalyzer
    analyzer = StatisticalAnalyzer()

    # Test with simple data
    amounts = [100, 200, 300, 400, 500]
    stats = analyzer.calculate_basic_statistics(amounts)

    assert 'mean' in stats
    assert 'median' in stats
    assert stats['mean'] == 300.0
    print("✅ Statistical analyzer test passed")

if __name__ == "__main__":
    print("🧪 Running Simple Tests...")
    print("=" * 40)

    test_basic_imports()
    test_agent_mock_imports()

    # Run async test
    asyncio.run(test_mock_agent_run())

    test_statistical_analyzer()

    print("=" * 40)
    print("🎉 ALL SIMPLE TESTS PASSED!")