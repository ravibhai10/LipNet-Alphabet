"""Run the supported single-process backend while holding a private-store lock."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.process_lock import acquire_process_lock, release_process_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')

    acquire_process_lock()
    try:
        import uvicorn
        uvicorn.run('app.main:app', host=args.host, port=args.port,
                    workers=1, reload=False)
    finally:
        release_process_lock()


if __name__ == '__main__':
    main()
