"""
Main entry point for alphabet package.

Allows: python -m alphabet <command>
"""

import sys

# Ensure Unicode output (check marks, emoji) works on Windows consoles that
# default to a legacy code page such as cp1252.
if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):  # pragma: no cover - unusual stream types
        pass
if sys.stderr is not None and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):  # pragma: no cover
        pass

from alphabet.cli import main

if __name__ == "__main__":
    sys.exit(main())
