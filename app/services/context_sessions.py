"""Server-authoritative interpretation sessions; independent of recording consent.

Submission fencing uses a process-local lock shared by the app's singleton store.
The consent/submission invariant therefore requires the existing single-worker,
single-process backend deployment; multiple worker processes need a shared fence.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import secrets
import sqlite3
import threading
import uuid
from pathlib import Path

from app.settings import settings

PURPOSE = 'text_only_llm_interpretation'
NOTICE_VERSION = 'llm-text-v1'
SCHEMA_VERSION = 1
SESSION_TTL = timedelta(hours=8)


def utcnow():
    return datetime.now(timezone.utc)


def stamp(value=None):
    return (value or utcnow()).isoformat()


class ContextSessionError(Exception):
    def __init__(self, code, status=403):
        self.code, self.status = code, status
        super().__init__(code)


class ContextSessionStore:
    """Uses the private research SQLite file, but requires explicit schema migration."""
    def __init__(self, db_path: str | Path | None = None, clock=utcnow):
        self.db_path = Path(db_path) if db_path else settings().private_root / 'feedback.sqlite3'
        self.clock = clock
        self._lock = threading.RLock()

    def _db(self):
        if self.db_path.is_symlink() or not self.db_path.is_file():
            raise ContextSessionError('session_store_unavailable', 503)
        db = sqlite3.connect(self.db_path, timeout=5, check_same_thread=False)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('PRAGMA secure_delete=ON')
        if db.execute('PRAGMA user_version').fetchone()[0] != SCHEMA_VERSION:
            db.close()
            raise ContextSessionError('session_migration_required', 503)
        return db

    def create(self, owner: str, participant_id: str | None = None):
        token = secrets.token_urlsafe(32)
        session_id = uuid.uuid4().hex
        created = self.clock()
        expires = created + SESSION_TTL
        with self._lock:
            db = self._db()
            try:
                db.execute('BEGIN IMMEDIATE')
                if participant_id is not None:
                    exists = db.execute('SELECT 1 FROM participants WHERE owner=? AND id=?',
                                        (owner, participant_id)).fetchone()
                    if not exists:
                        raise ContextSessionError('participant_not_verified_or_not_owned', 422)
                db.execute('INSERT INTO context_sessions(id,token_sha256,owner,participant_id,created_at,expires_at) VALUES(?,?,?,?,?,?)',
                           (session_id, hashlib.sha256(token.encode()).hexdigest(), owner, participant_id,
                            stamp(created), stamp(expires)))
                db.commit()
            except BaseException:
                db.rollback()
                raise
            finally:
                db.close()
        return {'session_id': session_id, 'session_token': token, 'expires_at': stamp(expires),
                'participant_bound': participant_id is not None}

    def _find(self, db, owner, token):
        digest = hashlib.sha256(token.encode()).hexdigest()
        row = db.execute('SELECT * FROM context_sessions WHERE token_sha256=?', (digest,)).fetchone()
        if row is None or row['owner'] != owner:
            raise ContextSessionError('session_not_found_or_not_owned', 401)
        if row['terminated_at'] is not None:
            raise ContextSessionError('session_terminated', 401)
        if datetime.fromisoformat(row['expires_at']) <= self.clock():
            raise ContextSessionError('session_expired', 401)
        return row

    def grant_consent(self, owner, token, purpose, notice_version, granted):
        if granted is not True or purpose != PURPOSE or notice_version != NOTICE_VERSION:
            raise ContextSessionError('explicit_current_consent_required', 422)
        with self._lock:
            db = self._db()
            try:
                db.execute('BEGIN IMMEDIATE')
                row = self._find(db, owner, token)
                db.execute('UPDATE context_session_consents SET revoked_at=? WHERE session_id=? AND purpose=? AND revoked_at IS NULL',
                           (stamp(self.clock()), row['id'], PURPOSE))
                db.execute('INSERT INTO context_session_consents(session_id,purpose,notice_version,granted_at,revoked_at) VALUES(?,?,?,?,NULL)',
                           (row['id'], PURPOSE, NOTICE_VERSION, stamp(self.clock())))
                consent_id=db.execute('SELECT last_insert_rowid()').fetchone()[0]
                db.commit()
                return {**self._scope(row),'consent_id':consent_id}
            except BaseException:
                db.rollback()
                raise
            finally:
                db.close()

    def revoke_consent(self, owner, token):
        with self._lock:
            db = self._db()
            try:
                db.execute('BEGIN IMMEDIATE')
                row = self._find(db, owner, token)
                db.execute('UPDATE context_session_consents SET revoked_at=? WHERE session_id=? AND purpose=? AND revoked_at IS NULL',
                           (stamp(self.clock()), row['id'], PURPOSE))
                db.commit()
                return self._scope(row)
            except BaseException:
                db.rollback()
                raise
            finally:
                db.close()

    def terminate(self, owner, token):
        with self._lock:
            db = self._db()
            try:
                db.execute('BEGIN IMMEDIATE')
                row = self._find(db, owner, token)
                db.execute('UPDATE context_session_consents SET revoked_at=? WHERE session_id=? AND revoked_at IS NULL',
                           (stamp(self.clock()), row['id']))
                db.execute('UPDATE context_sessions SET terminated_at=? WHERE id=?', (stamp(self.clock()), row['id']))
                db.commit()
                return self._scope(row)
            except BaseException:
                db.rollback()
                raise
            finally:
                db.close()

    def authorize(self, owner, token, require_consent=True):
        with self._lock:
            db = self._db()
            try:
                row = self._find(db, owner, token)
                consent = db.execute('SELECT id FROM context_session_consents WHERE session_id=? AND purpose=? AND notice_version=? AND revoked_at IS NULL ORDER BY id DESC LIMIT 1',
                                     (row['id'], PURPOSE, NOTICE_VERSION)).fetchone()
                if require_consent and consent is None:
                    raise ContextSessionError('consent_required', 403)
                return {**self._scope(row), 'consent_id': consent['id'] if consent else None}
            finally:
                db.close()

    def is_current(self, owner, scope, consent_id):
        try:
            current = self.authorize_scope(owner, scope['session_id'], consent_id)
            return current is not None
        except (ContextSessionError, OSError, sqlite3.Error):
            return False

    def commit_if_current(self, owner, scope, consent_id, operation):
        """Serialize cache publication with revocation/termination in this worker."""
        with self._lock:
            if self.authorize_scope(owner, scope['session_id'], consent_id) is None:
                return False
            operation()
            return True

    def submit_if_current(self, owner, scope, consent_id, submit):
        """Authorize and synchronously hand work to the provider as one fenced step.

        ``submit`` must cross the provider's irreversible submission boundary before
        returning (for the HTTP adapter, enqueueing the request in its transport
        executor). Revocation uses the same lock, so it either commits first and
        blocks this callback, or waits until the submission handoff has completed.
        No SQLite transaction is held while ``submit`` runs.
        """
        with self._lock:
            if self.authorize_scope(owner, scope['session_id'], consent_id) is None:
                return False, None
            return True, submit()

    def authorize_scope(self, owner, session_id, consent_id):
        with self._lock:
            db = self._db()
            try:
                row = db.execute('SELECT * FROM context_sessions WHERE id=? AND owner=?', (session_id, owner)).fetchone()
                if row is None or row['terminated_at'] is not None or datetime.fromisoformat(row['expires_at']) <= self.clock():
                    return None
                consent = db.execute('SELECT 1 FROM context_session_consents WHERE id=? AND session_id=? AND purpose=? AND notice_version=? AND revoked_at IS NULL',
                                     (consent_id, session_id, PURPOSE, NOTICE_VERSION)).fetchone()
                return self._scope(row) if consent else None
            finally:
                db.close()

    @staticmethod
    def _scope(row):
        return {'session_id': row['id'], 'participant_id': row['participant_id']}


context_session_store = ContextSessionStore()
