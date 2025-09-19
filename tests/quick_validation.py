#!/usr/bin/env python3
"""
Quick validation script for testing core components
"""

import os
import sys

def main():
    print("🔍 Quick Validation Test...")
    print("=" * 40)

    # Set test environment
    os.environ['TEST_MODE'] = 'true'
    os.environ['GROQ_API_KEY'] = 'test-key-for-validation'

    try:
        # Test 1: Config
        print("1. Testing configuration...")
        from backend.core.config import settings, get_settings
        config = get_settings()
        print(f"   ✅ GROQ_MODEL_MAIN: {config.GROQ_MODEL_MAIN}")
        print(f"   ✅ TEST_MODE: {config.TEST_MODE}")

        # Test 2: Basic imports
        print("\n2. Testing basic imports...")
        from backend.workflows.nodes import ingest_agent, TEST_MODE
        print(f"   ✅ Agent imported (TEST_MODE: {TEST_MODE})")

        # Test 3: Service import
        print("\n3. Testing service import...")
        from backend.services.audit_service import AuditService
        print("   ✅ AuditService imported")

        print("\n" + "=" * 40)
        print("🎉 QUICK VALIDATION PASSED!")
        return 0

    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())