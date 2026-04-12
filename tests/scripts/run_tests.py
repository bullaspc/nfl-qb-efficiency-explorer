#!/usr/bin/env python3
"""
run_tests.py — Test runner with regression detection for Streamlit apps.

Usage:
  python tests/scripts/run_tests.py                  # run and detect regressions
  python tests/scripts/run_tests.py --update-baseline  # run and save as new baseline
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BASELINE_FILE = Path(__file__).parent.parent / ".test_baseline.json"


def run_pytest(test_dir: Path) -> dict:
    """Run pytest and return structured results."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_dir), "-v", "--tb=short", "--no-header"],
        capture_output=True,
        text=True,
    )
    return parse_pytest_output(result.stdout + result.stderr, result.returncode)


def parse_pytest_output(output: str, returncode: int) -> dict:
    """Parse pytest -v output into structured pass/fail lists."""
    passed, failed, skipped = [], [], []
    current_failure = None
    failure_lines = []

    for line in output.splitlines():
        if " PASSED" in line:
            test_id = line.split(" PASSED")[0].strip()
            passed.append(test_id)
            if current_failure:
                failed.append({"id": current_failure, "detail": "\n".join(failure_lines).strip()})
                current_failure = None
                failure_lines = []
        elif " FAILED" in line:
            test_id = line.split(" FAILED")[0].strip()
            if current_failure:
                failed.append({"id": current_failure, "detail": "\n".join(failure_lines).strip()})
            current_failure = test_id
            failure_lines = []
        elif " SKIPPED" in line:
            test_id = line.split(" SKIPPED")[0].strip()
            skipped.append(test_id)
        elif current_failure and line.strip():
            failure_lines.append(line)

    if current_failure:
        failed.append({"id": current_failure, "detail": "\n".join(failure_lines).strip()})

    return {
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "returncode": returncode,
        "raw_output": output,
    }


def load_baseline() -> dict:
    if BASELINE_FILE.exists():
        return json.loads(BASELINE_FILE.read_text())
    return {"passed": [], "recorded_at": None}


def save_baseline(passed_ids: list[str]):
    data = {"passed": passed_ids, "recorded_at": datetime.utcnow().isoformat()}
    BASELINE_FILE.write_text(json.dumps(data, indent=2))
    print(f"Baseline saved ({len(passed_ids)} tests) → {BASELINE_FILE}")


def report(results: dict, baseline: dict):
    passed_ids = results["passed"]
    failed_ids = [f["id"] for f in results["failed"]]
    baseline_passed = set(baseline.get("passed", []))

    regressions = [f for f in results["failed"] if f["id"] in baseline_passed]
    new_failures = [f for f in results["failed"] if f["id"] not in baseline_passed]
    new_passes = [p for p in passed_ids if p not in baseline_passed and baseline_passed]

    total = len(passed_ids) + len(failed_ids) + len(results["skipped"])
    print(f"\nTest run: {len(passed_ids)} passed, {len(failed_ids)} failed, {len(results['skipped'])} skipped\n")

    if regressions:
        print("REGRESSIONS (used to pass, now broken):")
        for f in regressions:
            print(f"  ✗ {f['id']}")
            if f["detail"]:
                for line in f["detail"].splitlines()[:6]:
                    print(f"      {line}")
        print()

    if new_failures:
        print("NEW FAILURES (never passed):")
        for f in new_failures:
            print(f"  ✗ {f['id']}")
            if f["detail"]:
                for line in f["detail"].splitlines()[:6]:
                    print(f"      {line}")
        print()

    if new_passes:
        print(f"NEW PASSES ({len(new_passes)} tests now passing that weren't in baseline):")
        for p in new_passes:
            print(f"  ✓ {p}")
        print("  → Run with --update-baseline to record these.\n")

    if not regressions and not new_failures:
        if not baseline_passed:
            print("No baseline yet. Run with --update-baseline to record a clean baseline.")
        else:
            print("No regressions detected.")

    return bool(regressions or new_failures)


def main():
    parser = argparse.ArgumentParser(description="Run tests with regression detection")
    parser.add_argument("--update-baseline", action="store_true", help="Save current passing tests as baseline")
    parser.add_argument("--test-dir", default="tests", help="Directory containing tests (default: tests/)")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        print(f"Error: test directory '{test_dir}' not found. Create it first.", file=sys.stderr)
        sys.exit(1)

    print(f"Running tests in {test_dir.resolve()} ...")
    results = run_pytest(test_dir)

    if args.update_baseline:
        save_baseline(results["passed"])
        if results["failed"]:
            print(f"Warning: {len(results['failed'])} tests are failing — baseline only records passing tests.")
        sys.exit(results["returncode"])

    baseline = load_baseline()
    if baseline["recorded_at"]:
        print(f"Comparing against baseline from {baseline['recorded_at']}")
    else:
        print("No baseline found — running without regression comparison.")

    has_problems = report(results, baseline)
    sys.exit(1 if has_problems else 0)


if __name__ == "__main__":
    main()
