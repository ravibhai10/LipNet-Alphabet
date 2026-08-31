#!/usr/bin/env python
"""Run all tests directly without pytest subprocess."""

import sys
import traceback

def run_test_file(filepath, description):
    """Run a single test file."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"File: {filepath}")
    print(f"{'='*60}")

    try:
        # Execute the test file
        with open(filepath, 'r') as f:
            code = f.read()

        # Create a namespace for the test
        namespace = {'__name__': '__main__'}
        exec(code, namespace)

        print(f"✓ {description} - PASSED")
        return True

    except Exception as e:
        print(f"✗ {description} - FAILED")
        print(f"Error: {e}")
        traceback.print_exc()
        return False

def main():
    test_files = [
        ("tests/test_schema.py", "Schema Tests"),
        ("tests/test_manifest.py", "Manifest Tests"),
        ("tests/test_geometry.py", "Geometry Tests"),
        ("tests/test_motion.py", "Motion Tests"),
        ("tests/test_fusion.py", "Fusion Tests"),
        ("tests/test_validation.py", "Validation Tests"),
        ("tests/test_preprocess.py", "Preprocess Tests"),
        ("tests/test_integration.py", "Integration Tests"),
        ("tests/test_cli.py", "CLI Tests"),
    ]

    results = []
    for filepath, desc in test_files:
        try:
            result = run_test_file(filepath, desc)
            results.append((desc, result))
        except FileNotFoundError:
            print(f"\n⚠ {description} - SKIPPED (file not found: {filepath})")
            results.append((desc, False))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for desc, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {status}: {desc}")

    print(f"\nTotal: {passed}/{total} passed")

    if passed == total:
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n✗ {total - passed} TEST(S) FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())