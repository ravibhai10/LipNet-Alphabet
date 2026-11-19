"""Explicit, backup-first schema migration for interpretation sessions.

This migration is never run at application startup. Use ``python -m
pipeline.adaptive.context_sessions migrate [database]``; a verified SQLite
backup is created before changes. ``rollback`` is allowed only while no
sessions exist. ``restore database backup_file`` first preserves the current
database as another backup, then restores the selected prior snapshot.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

MIGRATION_VERSION = 1

SCHEMA = '''
CREATE TABLE context_sessions (
  id TEXT PRIMARY KEY,
  token_sha256 TEXT NOT NULL UNIQUE,
  owner TEXT NOT NULL,
  participant_id TEXT,
  created_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  terminated_at TEXT,
  FOREIGN KEY(owner, participant_id) REFERENCES participants(owner, id)
);
CREATE INDEX context_sessions_owner_expiry ON context_sessions(owner, expires_at);
CREATE TABLE context_session_consents (
  id INTEGER PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES context_sessions(id),
  purpose TEXT NOT NULL,
  notice_version TEXT NOT NULL,
  granted_at TEXT NOT NULL,
  revoked_at TEXT
);
CREATE INDEX context_session_consents_active ON context_session_consents(session_id, purpose, revoked_at);
'''


def _backup_path(db_path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    return db_path.with_name(f'{db_path.name}.pre-context-sessions-v{MIGRATION_VERSION}-{stamp}.bak')


def backup_database(db_path: Path) -> Path:
    if not db_path.is_file() or db_path.is_symlink():
        raise FileNotFoundError('A regular existing SQLite database is required.')
    target = _backup_path(db_path)
    source = sqlite3.connect(f'file:{db_path.resolve().as_posix()}?mode=ro', uri=True, timeout=5)
    backup = sqlite3.connect(target)
    try:
        source.backup(backup)
        backup.commit()
    except BaseException:
        backup.close()
        target.unlink(missing_ok=True)
        raise
    finally:
        source.close()
        try:
            backup.close()
        except Exception:
            pass
    return target


def migrate(db_path: str | Path) -> Path:
    path = Path(db_path).resolve()
    backup = backup_database(path)
    connection = sqlite3.connect(path, timeout=5)
    try:
        connection.execute('PRAGMA foreign_keys=ON')
        integrity=connection.execute('PRAGMA integrity_check').fetchone()[0]
        if integrity != 'ok':
            raise RuntimeError('Migration refused: source database failed SQLite integrity_check.')
        connection.execute('BEGIN IMMEDIATE')
        version = connection.execute('PRAGMA user_version').fetchone()[0]
        if version != 0:
            raise RuntimeError(f'Expected schema version 0; found {version}.')
        existing={row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {'participants','captures'} <= existing:
            raise RuntimeError('Expected private research tables are unavailable.')
        statement = ''
        for line in SCHEMA.splitlines():
            statement += line + '\n'
            if sqlite3.complete_statement(statement):
                if statement.strip():
                    connection.execute(statement)
                statement = ''
        if statement.strip():
            raise RuntimeError('Incomplete migration statement.')
        if connection.execute('PRAGMA foreign_key_check').fetchone() is not None:
            raise RuntimeError('Migration refused: foreign key validation failed.')
        connection.execute(f'PRAGMA user_version={MIGRATION_VERSION}')
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()
    return backup


def rollback(db_path: str | Path) -> Path:
    path = Path(db_path).resolve()
    backup = backup_database(path)
    connection = sqlite3.connect(path, timeout=5)
    try:
        connection.execute('PRAGMA foreign_keys=ON')
        connection.execute('BEGIN IMMEDIATE')
        version = connection.execute('PRAGMA user_version').fetchone()[0]
        if version != MIGRATION_VERSION:
            raise RuntimeError(f'Expected schema version {MIGRATION_VERSION}; found {version}.')
        count = connection.execute('SELECT COUNT(*) FROM context_sessions').fetchone()[0]
        if count:
            raise RuntimeError('Rollback refused: interpretation sessions exist. Preserve data and use a forward migration.')
        connection.execute('DROP TABLE context_session_consents')
        connection.execute('DROP TABLE context_sessions')
        connection.execute('PRAGMA user_version=0')
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()
    return backup


def restore_database(db_path: str | Path, backup_path: str | Path) -> Path:
    """Restore a SQLite backup transactionally, first preserving current state."""
    path, source_path = Path(db_path).resolve(), Path(backup_path).resolve()
    if path == source_path or not source_path.is_file() or source_path.is_symlink():
        raise ValueError('A separate regular SQLite backup file is required.')
    current_backup = backup_database(path)
    source = sqlite3.connect(f'file:{source_path.as_posix()}?mode=ro', uri=True, timeout=5)
    target = sqlite3.connect(path, timeout=5)
    try:
        integrity = source.execute('PRAGMA integrity_check').fetchone()[0]
        if integrity != 'ok':
            raise RuntimeError('Restore source failed SQLite integrity_check.')
        source.backup(target)
        target.commit()
        if target.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise RuntimeError('Restored database failed SQLite integrity_check.')
    except BaseException:
        target.rollback()
        raise
    finally:
        source.close()
        target.close()
    return current_backup


def main(argv=None):
    from app.settings import settings
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('migrate', 'rollback', 'restore'))
    parser.add_argument('database', nargs='?', type=Path, default=settings().private_root / 'feedback.sqlite3')
    parser.add_argument('backup_file', nargs='?', type=Path)
    args = parser.parse_args(argv)
    if args.operation == 'restore':
        if args.backup_file is None:
            parser.error('restore requires a backup_file argument')
        backup = restore_database(args.database, args.backup_file)
    else:
        backup = migrate(args.database) if args.operation == 'migrate' else rollback(args.database)
    print(f'{args.operation} complete; backup: {backup}')


if __name__ == '__main__':
    main()
