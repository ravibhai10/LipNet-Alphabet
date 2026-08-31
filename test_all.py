#!/usr/bin/env python
"""Run all tests in sequence."""

import sys
import os

# Add project root to path
sys.path.insert(0, r'C:\Users\ravik\OneDrive\Desktop\lip alphabet')

test_files = [
    "tests/test_schema.py",
    "tests/test_manifest.py",
    "tests/test_geometry.py",
    "tests/test_motion.py",
    "tests/test_fusion.py",
    "tests/test_validate.py",
    "tests/test_preprocess.py",
    "tests/test_integration.py",
    "tests/test_cli.py",
]

def run_test_file(filepath):
    """Run a single test file and return True if passed."""
    print(f"\n{'='*60}")
    print(f"Running: {filepath}")
    print(f"{'='*60}")

    try:
        with open(filepath, 'r') as f:
            code = f.read()

        namespace = {'__name__': '__main__', '__file__': filepath}
        exec(code, namespace)

        # Check if pytest was run
        if hasattr(namespace, 'pytest') and namespace.get('__name__') == '__main__':
            # pytest was already called in the file
            pass

        print(f"✓ PASSED: {filepath}")
        return True

    except SystemExit as e:
        if e.code == 0:
            print(f"✓ PASSED: {filepath} (pytest exited with 0)")
            return True
        else:
            print(f"✗ FAILED: {filepath} (pytest exited with {e.code})")
            return False
    except Exception as e:
        print(f"✗ FAILED: {filepath}")
        print(f"  Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("RUNNING ALL TESTS")
    print("="*60)

    passed = 0
    failed = 0

    for test_file in test_files:
        if os.path.exists(test_file):
            if run_test_file(test_file):
                passed += 1
            else:
                failed += 1
        else:
            print(f"\n⚠ SKIPPED: {test_file} (not found)")
            failed += 1

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")

    if failed == 0:
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n❌ {failed} TEST(S) FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())