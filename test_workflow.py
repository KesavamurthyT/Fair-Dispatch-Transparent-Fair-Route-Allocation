#!/usr/bin/env python
"""
Production QA Automation CLI for Fair Dispatch LangGraph Backend.

Validates the ENTIRE LangGraph-migrated system:
- 5 Agent Nodes (ML Effort, Route Planner, Fairness, Liaison, Explainability)
- Gemini 3 Flash explanations
- Phases 1-8 functionality
- Performance requirements

Usage:
    python test_workflow.py --help
    python test_workflow.py --full-e2e
    python test_workflow.py --ev-stress
    python test_workflow.py --recovery-stress
    python test_workflow.py --gemini-only
    python test_workflow.py --timeline-validate
    python test_workflow.py --all
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import date, datetime
from typing import Dict, List, Any, Optional
from uuid import uuid4

try:
    import httpx
except ImportError:
    print("Error: httpx not installed. Run: pip install httpx")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

API_BASE_URL = "http://localhost:8000/api/v1"
LANGGRAPH_ENDPOINT = f"{API_BASE_URL}/allocate/langgraph"
ORIGINAL_ENDPOINT = f"{API_BASE_URL}/allocate"

# Performance thresholds
MAX_DURATION_50_DRIVERS = 20.0  # seconds
MAX_DURATION_10_DRIVERS = 5.0   # seconds


# =============================================================================
# TEST DATA FIXTURES
# =============================================================================

def generate_drivers(count: int, ev_ratio: float = 0.2, high_debt_ratio: float = 0.3) -> List[Dict]:
    """Generate realistic driver test data."""
    drivers = []
    for i in range(count):
        is_ev = i < int(count * ev_ratio)
        has_high_debt = i < int(count * high_debt_ratio)
        
        drivers.append({
            "id": f"drv_{i+1:03d}",
            "name": f"Driver {i+1}",
            "vehicle_capacity_kg": 80.0 if is_ev else 120.0,
            "preferred_language": "ta" if i % 3 == 0 else "en",
        })
    return drivers


def generate_packages(count: int, warehouse_lat: float = 13.0827, warehouse_lng: float = 80.2707) -> List[Dict]:
    """Generate realistic package test data."""
    packages = []
    for i in range(count):
        # Spread packages in 10km radius around warehouse
        lat_offset = (i % 10 - 5) * 0.01
        lng_offset = (i // 10 % 10 - 5) * 0.01
        
        packages.append({
            "id": f"pkg_{i+1:04d}",
            "weight_kg": 2.0 + (i % 10) * 0.5,
            "fragility_level": (i % 5) + 1,
            "address": f"Address {i+1}, Chennai",
            "latitude": warehouse_lat + lat_offset,
            "longitude": warehouse_lng + lng_offset,
            "priority": ["NORMAL", "NORMAL", "EXPRESS", "NORMAL", "HIGH"][i % 5],
        })
    return packages


def create_allocation_request(
    num_drivers: int = 10,
    num_packages: int = 50,
    ev_ratio: float = 0.2,
    allocation_date: str = None,
) -> Dict:
    """Create a complete allocation request."""
    if allocation_date is None:
        allocation_date = date.today().isoformat()
    
    return {
        "allocation_date": allocation_date,
        "drivers": generate_drivers(num_drivers, ev_ratio=ev_ratio),
        "packages": generate_packages(num_packages),
        "warehouse": {
            "lat": 13.0827,
            "lng": 80.2707,
        }
    }


# Pre-defined test scenarios
TEST_INPUTS = {
    "full_e2e": create_allocation_request(num_drivers=50, num_packages=250, ev_ratio=0.2),
    "small": create_allocation_request(num_drivers=5, num_packages=25, ev_ratio=0.2),
    "medium": create_allocation_request(num_drivers=10, num_packages=50, ev_ratio=0.2),
    "ev_stress": create_allocation_request(num_drivers=20, num_packages=100, ev_ratio=0.5),
    "recovery_stress": create_allocation_request(num_drivers=15, num_packages=75, ev_ratio=0.1),
}


# =============================================================================
# GOLDEN OUTPUTS (Expected Responses)
# =============================================================================

GOLDEN_OUTPUTS = {
    "full_e2e": {
        "status_code": 200,
        "gini_index": {"min": 0.15, "max": 0.45},
        "num_assignments": 50,
        "max_duration_s": 25.0,
        "required_fields": ["allocation_run_id", "allocation_date", "global_fairness", "assignments"],
        "timeline_agents": ["ML_EFFORT", "ROUTE_PLANNER", "FAIRNESS_MANAGER"],
    },
    "small": {
        "status_code": 200,
        "gini_index": {"min": 0.10, "max": 0.50},
        "num_assignments": 5,
        "max_duration_s": 5.0,
    },
    "medium": {
        "status_code": 200,
        "gini_index": {"min": 0.10, "max": 0.50},
        "num_assignments": 10,
        "max_duration_s": 10.0,
    },
}


# =============================================================================
# TEST RUNNER
# =============================================================================

class TestResult:
    """Container for test results."""
    def __init__(self, name: str):
        self.name = name
        self.passed = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.duration_s: float = 0
        self.data: Dict = {}
    
    def fail(self, message: str):
        self.passed = False
        self.errors.append(message)
    
    def warn(self, message: str):
        self.warnings.append(message)
    
    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        msg = f"{status} {self.name} ({self.duration_s:.2f}s)"
        for err in self.errors:
            msg += f"\n    ❌ {err}"
        for warn in self.warnings:
            msg += f"\n    ⚠️  {warn}"
        # Show error response if available
        if not self.passed and self.data.get("error"):
            msg += f"\n    📋 Response: {self.data['error'][:300]}"
        return msg



async def run_allocation_test(
    test_name: str,
    request_data: Dict,
    golden: Dict,
    endpoint: str = LANGGRAPH_ENDPOINT,
    enable_gemini: bool = False,
) -> TestResult:
    """Run a single allocation test."""
    result = TestResult(test_name)
    
    url = f"{endpoint}?enable_gemini={str(enable_gemini).lower()}"
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            start = time.time()
            response = await client.post(url, json=request_data)
            result.duration_s = time.time() - start
            
            # Status code check
            if response.status_code != golden.get("status_code", 200):
                result.fail(f"Status code {response.status_code}, expected {golden['status_code']}")
                result.data["error"] = response.text[:500]
                return result
            
            data = response.json()
            result.data = data
            
            # Required fields check
            for field in golden.get("required_fields", ["allocation_run_id", "assignments"]):
                if field not in data:
                    result.fail(f"Missing required field: {field}")
            
            # Assignments count check
            if "num_assignments" in golden:
                actual = len(data.get("assignments", []))
                expected = golden["num_assignments"]
                if actual != expected:
                    result.fail(f"Assignment count {actual}, expected {expected}")
            
            # Gini index check
            if "gini_index" in golden:
                gini = data.get("global_fairness", {}).get("gini_index", 0)
                if not (golden["gini_index"]["min"] <= gini <= golden["gini_index"]["max"]):
                    result.warn(f"Gini {gini:.3f} outside expected range [{golden['gini_index']['min']}, {golden['gini_index']['max']}]")
            
            # Performance check
            if "max_duration_s" in golden:
                if result.duration_s > golden["max_duration_s"]:
                    result.fail(f"Duration {result.duration_s:.2f}s exceeds max {golden['max_duration_s']}s")
            
            # Gemini check
            if enable_gemini:
                for assignment in data.get("assignments", []):
                    explanation = assignment.get("explanation", "")
                    if len(explanation) < 10:
                        result.warn(f"Short explanation for {assignment.get('driver_id')}")
                        break
            
    except httpx.ConnectError:
        result.fail("Cannot connect to server. Is uvicorn running?")
    except httpx.TimeoutException:
        result.fail(f"Request timed out after 60s")
    except Exception as e:
        result.fail(f"Exception: {str(e)[:200]}")
    
    return result


# =============================================================================
# TEST SUITES
# =============================================================================

async def test_full_e2e() -> TestResult:
    """Full end-to-end test with 50 drivers."""
    print("\n🧪 Running Full E2E Test (50 drivers, 250 packages)...")
    return await run_allocation_test(
        "Full E2E",
        TEST_INPUTS["full_e2e"],
        GOLDEN_OUTPUTS["full_e2e"],
    )


async def test_small() -> TestResult:
    """Quick sanity test with 5 drivers."""
    print("\n🧪 Running Small Test (5 drivers)...")
    return await run_allocation_test(
        "Small",
        TEST_INPUTS["small"],
        GOLDEN_OUTPUTS["small"],
    )


async def test_medium() -> TestResult:
    """Medium test with 10 drivers."""
    print("\n🧪 Running Medium Test (10 drivers)...")
    return await run_allocation_test(
        "Medium",
        TEST_INPUTS["medium"],
        GOLDEN_OUTPUTS["medium"],
    )


async def test_ev_stress() -> TestResult:
    """EV stress test with 50% EV drivers."""
    print("\n🧪 Running EV Stress Test (50% EV drivers)...")
    return await run_allocation_test(
        "EV Stress",
        TEST_INPUTS["ev_stress"],
        {"status_code": 200, "num_assignments": 20, "max_duration_s": 15.0},
    )


async def test_recovery_stress() -> TestResult:
    """Recovery stress test."""
    print("\n🧪 Running Recovery Stress Test...")
    return await run_allocation_test(
        "Recovery Stress",
        TEST_INPUTS["recovery_stress"],
        {"status_code": 200, "num_assignments": 15, "max_duration_s": 15.0},
    )


async def test_gemini_explanations() -> TestResult:
    """Test Gemini-powered explanations."""
    print("\n🧪 Running Gemini Explanations Test...")
    result = await run_allocation_test(
        "Gemini Explanations",
        TEST_INPUTS["small"],
        {"status_code": 200, "num_assignments": 5, "max_duration_s": 30.0},
        enable_gemini=True,
    )
    
    # Additional Gemini-specific validations
    if result.passed and result.data:
        languages_seen = set()
        for assignment in result.data.get("assignments", []):
            explanation = assignment.get("explanation", "")
            if explanation:
                # Check if Tamil characters present
                if any('\u0B80' <= c <= '\u0BFF' for c in explanation):
                    languages_seen.add("ta")
                else:
                    languages_seen.add("en")
        
        result.data["languages_detected"] = list(languages_seen)
        print(f"    Languages detected: {languages_seen}")
    
    return result


async def test_api_equivalence() -> TestResult:
    """Compare LangGraph vs Original endpoint responses."""
    print("\n🧪 Running API Equivalence Test...")
    result = TestResult("API Equivalence")
    
    request = TEST_INPUTS["small"]
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # LangGraph endpoint
            start1 = time.time()
            resp1 = await client.post(LANGGRAPH_ENDPOINT, json=request)
            time1 = time.time() - start1
            
            # Original endpoint
            start2 = time.time()
            resp2 = await client.post(ORIGINAL_ENDPOINT, json=request)
            time2 = time.time() - start2
            
            result.duration_s = time1 + time2
            
            if resp1.status_code != resp2.status_code:
                result.fail(f"Status mismatch: LangGraph={resp1.status_code}, Original={resp2.status_code}")
                return result
            
            data1 = resp1.json()
            data2 = resp2.json()
            
            # Compare structure
            if set(data1.keys()) != set(data2.keys()):
                result.warn(f"Response keys differ: {set(data1.keys())} vs {set(data2.keys())}")
            
            # Compare assignment counts
            if len(data1.get("assignments", [])) != len(data2.get("assignments", [])):
                result.fail(f"Assignment count mismatch: {len(data1['assignments'])} vs {len(data2['assignments'])}")
            
            print(f"    LangGraph: {time1:.2f}s, Original: {time2:.2f}s")
            result.data = {"langgraph_time": time1, "original_time": time2}
            
    except Exception as e:
        result.fail(f"Exception: {str(e)[:200]}")
    
    return result


async def test_timeline_validate() -> TestResult:
    """Validate Phase 5 decision timeline."""
    print("\n🧪 Running Timeline Validation Test...")
    result = TestResult("Timeline Validation")
    
    # This would require querying the database for DecisionLog entries
    # For now, we verify the allocation completes successfully
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            start = time.time()
            resp = await client.post(LANGGRAPH_ENDPOINT, json=TEST_INPUTS["small"])
            result.duration_s = time.time() - start
            
            if resp.status_code == 200:
                data = resp.json()
                allocation_id = data.get("allocation_run_id")
                
                # Query timeline endpoint if available
                timeline_resp = await client.get(f"{API_BASE_URL}/admin/runs/{allocation_id}/timeline")
                if timeline_resp.status_code == 200:
                    timeline = timeline_resp.json()
                    agents = [entry.get("agent_name") for entry in timeline.get("timeline", [])]
                    
                    expected_agents = ["ML_EFFORT", "ROUTE_PLANNER", "FAIRNESS_MANAGER"]
                    for agent in expected_agents:
                        if agent not in agents:
                            result.warn(f"Expected agent {agent} not in timeline")
                    
                    result.data = {"timeline_agents": agents}
                    print(f"    Timeline agents: {agents}")
                else:
                    result.warn("Timeline endpoint not available (may need admin access)")
            else:
                result.fail(f"Allocation failed: {resp.status_code}")
                
    except Exception as e:
        result.fail(f"Exception: {str(e)[:200]}")
    
    return result


async def test_health_check() -> TestResult:
    """Basic health check."""
    print("\n🧪 Running Health Check...")
    result = TestResult("Health Check")
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            start = time.time()
            resp = await client.get(f"{API_BASE_URL.replace('/api/v1', '')}/health")
            result.duration_s = time.time() - start
            
            if resp.status_code != 200:
                result.fail(f"Health check failed: {resp.status_code}")
            else:
                result.data = resp.json()
                print(f"    Status: {result.data}")
                
    except httpx.ConnectError:
        result.fail("Cannot connect to server. Is uvicorn running?")
    except Exception as e:
        result.fail(f"Exception: {str(e)}")
    
    return result


# =============================================================================
# CLI INTERFACE
# =============================================================================

async def run_tests(args) -> int:
    """Run selected tests based on CLI arguments."""
    results: List[TestResult] = []
    
    # Always run health check first
    health = await test_health_check()
    results.append(health)
    
    if not health.passed:
        print("\n❌ Health check failed. Is the server running?")
        print("   Start with: uvicorn app.main:app --reload")
        return 1
    
    # Run selected tests
    if args.all:
        results.append(await test_small())
        results.append(await test_medium())
        results.append(await test_api_equivalence())
        results.append(await test_ev_stress())
        results.append(await test_timeline_validate())
        if args.gemini:
            results.append(await test_gemini_explanations())
        if args.full:
            results.append(await test_full_e2e())
    else:
        if args.full_e2e:
            results.append(await test_full_e2e())
        if args.small:
            results.append(await test_small())
        if args.medium:
            results.append(await test_medium())
        if args.ev_stress:
            results.append(await test_ev_stress())
        if args.recovery_stress:
            results.append(await test_recovery_stress())
        if args.gemini_only:
            results.append(await test_gemini_explanations())
        if args.timeline_validate:
            results.append(await test_timeline_validate())
        if args.equivalence:
            results.append(await test_api_equivalence())
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    for r in results:
        print(r)
        if r.passed:
            passed += 1
        else:
            failed += 1
    
    print("=" * 60)
    print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed}")
    
    if failed == 0:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed.")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Fair Dispatch LangGraph Backend QA Automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_workflow.py --small         # Quick 5-driver test
  python test_workflow.py --medium        # 10-driver test
  python test_workflow.py --full-e2e      # Full 50-driver test
  python test_workflow.py --all           # Run all tests
  python test_workflow.py --gemini-only   # Test Gemini explanations
  python test_workflow.py --equivalence   # Compare LangGraph vs Original
        """
    )
    
    parser.add_argument("--full-e2e", action="store_true", help="Full E2E test (50 drivers)")
    parser.add_argument("--small", action="store_true", help="Quick sanity test (5 drivers)")
    parser.add_argument("--medium", action="store_true", help="Medium test (10 drivers)")
    parser.add_argument("--ev-stress", action="store_true", help="EV stress test (50%% EV)")
    parser.add_argument("--recovery-stress", action="store_true", help="Recovery stress test")
    parser.add_argument("--gemini-only", action="store_true", help="Test Gemini explanations")
    parser.add_argument("--timeline-validate", action="store_true", help="Validate Phase 5 timeline")
    parser.add_argument("--equivalence", action="store_true", help="Compare LangGraph vs Original")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--gemini", action="store_true", help="Include Gemini tests in --all")
    parser.add_argument("--full", action="store_true", help="Include full E2E in --all")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="API base URL")
    
    args = parser.parse_args()
    
    # Update URL if provided
    global API_BASE_URL, LANGGRAPH_ENDPOINT, ORIGINAL_ENDPOINT
    if args.url != "http://localhost:8000":
        API_BASE_URL = f"{args.url}/api/v1"
        LANGGRAPH_ENDPOINT = f"{API_BASE_URL}/allocate/langgraph"
        ORIGINAL_ENDPOINT = f"{API_BASE_URL}/allocate"
    
    # Default to --small if no tests specified
    if not any([args.full_e2e, args.small, args.medium, args.ev_stress, 
                args.recovery_stress, args.gemini_only, args.timeline_validate,
                args.equivalence, args.all]):
        args.small = True
    
    print("=" * 60)
    print("Fair Dispatch LangGraph QA Automation")
    print(f"Target: {API_BASE_URL}")
    print("=" * 60)
    
    exit_code = asyncio.run(run_tests(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
