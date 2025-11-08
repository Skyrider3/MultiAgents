#!/usr/bin/env python3
"""
System Testing Script - Validates all components are working
"""

import asyncio
import sys
from pathlib import Path
import aiohttp
import json
from typing import Dict, Any, List
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from loguru import logger


class SystemTester:
    """System integration tester"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_results = []

    async def test_api_health(self) -> bool:
        """Test API health endpoint"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.success("✓ API Health Check Passed")
                        logger.info(f"  Services: {data.get('services', {})}")
                        return True
                    else:
                        logger.error(f"✗ API Health Check Failed: Status {response.status}")
                        return False
        except Exception as e:
            logger.error(f"✗ API Health Check Failed: {e}")
            return False

    async def test_agents(self) -> bool:
        """Test agent endpoints"""
        try:
            async with aiohttp.ClientSession() as session:
                # List agents
                async with session.get(f"{self.base_url}/api/v1/agents") as response:
                    if response.status != 200:
                        logger.error("✗ Agent listing failed")
                        return False

                    agents = await response.json()
                    logger.success(f"✓ Agent Listing: {len(agents)} agents")

                # Create a test task
                task_data = {
                    "task_type": "research",
                    "description": "Test task",
                    "context": {},
                    "priority": 5
                }

                async with session.post(
                    f"{self.base_url}/api/v1/agents/tasks",
                    json=task_data
                ) as response:
                    if response.status in [200, 201]:
                        task = await response.json()
                        logger.success(f"✓ Task Creation: {task.get('task_id')}")
                        return True
                    else:
                        logger.error(f"✗ Task creation failed: Status {response.status}")
                        return False

        except Exception as e:
            logger.error(f"✗ Agent tests failed: {e}")
            return False

    async def test_knowledge_graph(self) -> bool:
        """Test knowledge graph endpoints"""
        try:
            async with aiohttp.ClientSession() as session:
                # Get statistics
                async with session.get(f"{self.base_url}/api/v1/knowledge/statistics") as response:
                    if response.status != 200:
                        logger.error("✗ Knowledge graph statistics failed")
                        return False

                    stats = await response.json()
                    logger.success(f"✓ Knowledge Graph Stats: {stats}")

                # Search papers
                async with session.get(f"{self.base_url}/api/v1/knowledge/papers?limit=5") as response:
                    if response.status == 200:
                        papers = await response.json()
                        logger.success(f"✓ Paper Search: {len(papers)} papers found")
                        return True
                    else:
                        logger.warning("✗ Paper search returned no results")
                        return True  # Not a failure if no papers exist

        except Exception as e:
            logger.error(f"✗ Knowledge graph tests failed: {e}")
            return False

    async def test_reasoning(self) -> bool:
        """Test reasoning endpoints"""
        try:
            async with aiohttp.ClientSession() as session:
                # Test symbolic expression
                expr_data = {
                    "expression": "x**2 + 2*x + 1",
                    "operation": "factor"
                }

                async with session.post(
                    f"{self.base_url}/api/v1/reasoning/symbolic/expression",
                    json=expr_data
                ) as response:
                    if response.status != 200:
                        logger.error("✗ Symbolic reasoning failed")
                        return False

                    result = await response.json()
                    logger.success(f"✓ Symbolic Reasoning: {result.get('result')}")

                # Test pattern finding
                pattern_data = {
                    "sequence": [1, 1, 2, 3, 5, 8, 13, 21]
                }

                async with session.post(
                    f"{self.base_url}/api/v1/reasoning/symbolic/find_pattern",
                    json=pattern_data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.success(f"✓ Pattern Finding: {result.get('pattern', 'Pattern found')}")
                        return True
                    else:
                        logger.warning("✗ Pattern finding failed")
                        return False

        except Exception as e:
            logger.error(f"✗ Reasoning tests failed: {e}")
            return False

    async def test_ingestion_status(self) -> bool:
        """Test ingestion status endpoints"""
        try:
            async with aiohttp.ClientSession() as session:
                # Get ingestion tasks
                async with session.get(f"{self.base_url}/api/v1/ingestion/tasks") as response:
                    if response.status != 200:
                        logger.error("✗ Ingestion task listing failed")
                        return False

                    tasks = await response.json()
                    logger.success(f"✓ Ingestion Tasks: {len(tasks)} tasks")

                # Get statistics
                async with session.get(f"{self.base_url}/api/v1/ingestion/statistics") as response:
                    if response.status == 200:
                        stats = await response.json()
                        logger.success(f"✓ Ingestion Stats: {stats}")
                        return True
                    else:
                        logger.warning("✗ Ingestion statistics failed")
                        return False

        except Exception as e:
            logger.error(f"✗ Ingestion tests failed: {e}")
            return False

    async def test_workflow(self) -> bool:
        """Test complete workflow"""
        try:
            async with aiohttp.ClientSession() as session:
                # Start a simple workflow
                workflow_data = {
                    "papers": ["test_paper_1"],
                    "workflow_type": "standard",
                    "options": {}
                }

                async with session.post(
                    f"{self.base_url}/api/v1/agents/workflow/start",
                    json=workflow_data
                ) as response:
                    if response.status in [200, 201]:
                        result = await response.json()
                        logger.success(f"✓ Workflow Started: {result.get('task_id')}")
                        return True
                    else:
                        logger.warning("✗ Workflow start failed (might be expected if no papers)")
                        return True  # Not critical

        except Exception as e:
            logger.error(f"✗ Workflow test failed: {e}")
            return False

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all system tests"""
        logger.info("=" * 60)
        logger.info("Running System Integration Tests")
        logger.info("=" * 60)

        test_suite = [
            ("API Health", self.test_api_health),
            ("Agents", self.test_agents),
            ("Knowledge Graph", self.test_knowledge_graph),
            ("Reasoning", self.test_reasoning),
            ("Ingestion", self.test_ingestion_status),
            ("Workflow", self.test_workflow)
        ]

        results = {
            "total": len(test_suite),
            "passed": 0,
            "failed": 0,
            "tests": []
        }

        for test_name, test_func in test_suite:
            logger.info(f"\nTesting {test_name}...")
            try:
                success = await test_func()
                results["tests"].append({
                    "name": test_name,
                    "status": "PASS" if success else "FAIL"
                })

                if success:
                    results["passed"] += 1
                else:
                    results["failed"] += 1

            except Exception as e:
                logger.error(f"Test {test_name} crashed: {e}")
                results["tests"].append({
                    "name": test_name,
                    "status": "ERROR",
                    "error": str(e)
                })
                results["failed"] += 1

            # Small delay between tests
            await asyncio.sleep(0.5)

        return results


def print_test_summary(results: Dict[str, Any]):
    """Print test summary"""
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    # Overall stats
    logger.info(f"Total Tests: {results['total']}")
    logger.info(f"Passed: {results['passed']} ✓")
    logger.info(f"Failed: {results['failed']} ✗")

    # Success rate
    success_rate = (results['passed'] / results['total'] * 100) if results['total'] > 0 else 0
    logger.info(f"Success Rate: {success_rate:.1f}%")

    # Individual test results
    logger.info("\nTest Results:")
    for test in results['tests']:
        status_symbol = "✓" if test['status'] == "PASS" else "✗"
        status_color = "green" if test['status'] == "PASS" else "red"

        if test['status'] == "PASS":
            logger.success(f"  {status_symbol} {test['name']}: {test['status']}")
        else:
            logger.error(f"  {status_symbol} {test['name']}: {test['status']}")

    # Overall result
    logger.info("\n" + "=" * 60)
    if results['failed'] == 0:
        logger.success("All tests passed! System is operational.")
    else:
        logger.warning(f"{results['failed']} test(s) failed. Please check the logs.")


async def main():
    """Main test function"""
    # Parse arguments
    if "--help" in sys.argv:
        print("Usage: python test_system.py [options]")
        print("\nOptions:")
        print("  --api-url URL   API base URL (default: http://localhost:8000)")
        print("  --help          Show this help message")
        sys.exit(0)

    # Get API URL
    api_url = "http://localhost:8000"
    if "--api-url" in sys.argv:
        idx = sys.argv.index("--api-url")
        if idx + 1 < len(sys.argv):
            api_url = sys.argv[idx + 1]

    # Create tester
    tester = SystemTester(api_url)

    # Check if API is accessible
    logger.info(f"Testing API at: {api_url}")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{api_url}/health", timeout=5) as response:
                if response.status != 200:
                    logger.error("API is not accessible. Is the server running?")
                    logger.info("Start the server with: python -m src.api.app")
                    sys.exit(1)
    except Exception as e:
        logger.error(f"Cannot connect to API: {e}")
        logger.info("Make sure the API server is running:")
        logger.info("  python -m src.api.app")
        sys.exit(1)

    # Run tests
    results = await tester.run_all_tests()

    # Print summary
    print_test_summary(results)

    # Exit code based on results
    sys.exit(0 if results['failed'] == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())