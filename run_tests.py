#!/usr/bin/env python
"""Run all tests programmatically."""

import sys
import pytest

# Run all tests
exit_code = pytest.main([
    "tests/",
    "-v",
    "--tb=short",
    "-x"  # Stop on first failure
])

sys.exit(exit_code)