import os
from pathlib import Path
import subprocess
import sys
import uuid


ROOT = Path(__file__).resolve().parents[1]
LAUNCH_PYTHON = (ROOT / '.venv-avhubert-cpu' / 'Scripts' / 'python.exe'
                 if (ROOT / '.venv-avhubert-cpu' / 'Scripts' / 'python.exe').is_file()
                 else Path(sys.executable))


def private_test_root():
    name = f'.backend-lock-test-{uuid.uuid4().hex}'
    path = ROOT / 'data' / 'adaptive_private' / name
    path.mkdir()
    return path, Path('data') / 'adaptive_private' / name


def test_process_lock_blocks_a_second_process_and_releases_on_exit():
    private_path, relative_path = private_test_root()
    env = os.environ.copy()
    env['LIPNET_PRIVATE_ROOT'] = str(relative_path)
    snippet = (
        "import runpy; m=runpy.run_path('scripts/run_backend.py'); "
        "lock=m['acquire_process_lock'](); print('LOCKED', flush=True); "
        "input(); lock.close()"
    )
    holder = subprocess.Popen(
        [str(LAUNCH_PYTHON), '-c', snippet], cwd=ROOT, env=env,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == 'LOCKED'
        contender = subprocess.run(
            [str(LAUNCH_PYTHON), '-c',
             "import asyncio; from app.main import preload_selected_models; "
             "asyncio.run(preload_selected_models())"],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=15,
        )
        assert contender.returncode != 0
        assert 'deployment lock' in contender.stderr
        holder.stdin.write('\n')
        holder.stdin.flush()
        assert holder.wait(timeout=10) == 0
        retry = subprocess.run(
            [str(LAUNCH_PYTHON), '-c',
             "import runpy; l=runpy.run_path('scripts/run_backend.py')['acquire_process_lock'](); l.close()"],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=15,
        )
        assert retry.returncode == 0, retry.stderr
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=10)
        for child in private_path.iterdir():
            child.unlink()
        private_path.rmdir()


def test_runner_configures_exactly_one_worker_without_reload(monkeypatch):
    import importlib.util
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location(
        'lipnet_run_backend', ROOT / 'scripts' / 'run_backend.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    fake_uvicorn = SimpleNamespace(run=lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setitem(sys.modules, 'uvicorn', fake_uvicorn)
    monkeypatch.setattr(module, 'acquire_process_lock', lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(sys, 'argv', ['run_backend.py', '--host', '127.0.0.1', '--port', '8765'])

    module.main()

    assert calls == [(('app.main:app',), {
        'host': '127.0.0.1', 'port': 8765, 'workers': 1, 'reload': False,
    })]


def test_session_store_can_read_write_private_database_copy():
    import sqlite3
    from app.services.context_sessions import ContextSessionStore, SCHEMA_VERSION

    private = ROOT / 'data' / 'adaptive_private'
    copy_path = private / f'.sqlite-acl-test-{uuid.uuid4().hex}.sqlite3'
    source_uri = f"file:{(private / 'feedback.sqlite3').as_posix()}?mode=ro"
    source = sqlite3.connect(source_uri, uri=True)
    destination = sqlite3.connect(copy_path)
    try:
        source.backup(destination)
        destination.execute(f'PRAGMA user_version={SCHEMA_VERSION}')
        destination.commit()
    finally:
        destination.close()
        source.close()

    try:
        store = ContextSessionStore(db_path=copy_path)
        db = store._db()
        try:
            db.execute('CREATE TABLE acl_probe (value TEXT NOT NULL)')
            db.execute("INSERT INTO acl_probe(value) VALUES ('write-ok')")
            db.commit()
            assert db.execute('SELECT value FROM acl_probe').fetchone()[0] == 'write-ok'
        finally:
            db.close()
    finally:
        copy_path.unlink(missing_ok=True)
        for suffix in ('-wal', '-shm'):
            Path(str(copy_path) + suffix).unlink(missing_ok=True)
