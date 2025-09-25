#!/usr/bin/env python3
"""
Docker-compatible Account Service Connection Test
This script runs the account service connection tests within the Docker Compose environment.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the backend src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from simple_test import simple_account_test
from health_test import test_health_endpoints
from common.logging import get_logger

logger = get_logger("account_docker_test")

async def run_docker_tests():
    """Run all account service tests in Docker environment"""
    print("=" * 60)
    print("ACCOUNT SERVICE DOCKER TESTS")
    print("=" * 60)
    
    # Print environment info
    print("\nEnvironment Information:")
    print(f"TWS_HOST: {os.getenv('TWS_HOST', 'Not set')}")
    print(f"TWS_PORT: {os.getenv('TWS_PORT', 'Not set')}")
    print(f"USE_PAPER: {os.getenv('USE_PAPER', 'Not set')}")
    print(f"ENABLE_LIVE: {os.getenv('ENABLE_LIVE', 'Not set')}")
    print(f"DRY_RUN: {os.getenv('DRY_RUN', 'Not set')}")
    
    tests = [
        ("Simple Connection Test", simple_account_test),
        ("Health Check Test", test_health_endpoints),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 40}")
        print(f"RUNNING: {test_name}")
        print('=' * 40)
        
        try:
            success = await test_func()
            results[test_name] = success
            
            if success:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
                
        except KeyboardInterrupt:
            print(f"\n🛑 Test interrupted by user")
            break
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            results[test_name] = False
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print('=' * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for success in results.values() if success)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nResults: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Account service connection is working.")
        print("✅ Ready to proceed with database implementation.")
    else:
        print("⚠️  Some tests failed. Please check TWS connection and configuration.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    try:
        success = asyncio.run(run_docker_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Test runner failed: {e}")
        sys.exit(1)
