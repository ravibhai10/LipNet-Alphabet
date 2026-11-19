import hashlib
import json
import sqlite3

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import app.security as security
import scripts.configure_local_access as credentials_cli
from app import context_routes
from app.services.context_sessions import ContextSessionStore, NOTICE_VERSION, PURPOSE
from pipeline.adaptive.context_sessions import migrate


TEST_TOKEN = 'fixture-runtime-test-token-' + 't' * 40
OTHER_TOKEN = 'fixture-other-collector-token-' + 'c' * 40


def _access_file(path, rows=None):
    rows = rows or [
        ('runtime-test', 'context_tester', TEST_TOKEN),
        ('other-collector', 'collector', OTHER_TOKEN),
    ]
    principals = [{'id': actor, 'role': role,
                   'token_sha256': [hashlib.sha256(token.encode()).hexdigest()]}
                  for actor, role, token in rows]
    path.write_text(json.dumps({'principals': principals}), encoding='utf-8')
    return path


def test_context_tester_bearer_and_actor_come_from_server_config(monkeypatch, tmp_path):
    config = _access_file(tmp_path / 'access.json')
    monkeypatch.setenv('LIPNET_ACCESS_FILE', str(config))
    result = security.access('Bearer ' + TEST_TOKEN)
    assert result == {'actor': 'runtime-test', 'role': 'context_tester'}
    assert security.access('Bearer ' + OTHER_TOKEN) == {'actor': 'other-collector', 'role': 'collector'}
    with pytest.raises(HTTPException) as invalid:
        security.access('Bearer ' + 'x' * len(TEST_TOKEN))
    assert invalid.value.status_code == 401
    with pytest.raises(HTTPException) as missing:
        security.access(None)
    assert missing.value.status_code == 401


def test_context_tester_digest_comparison_is_constant_time(monkeypatch, tmp_path):
    config = _access_file(tmp_path / 'access.json', [('runtime-test', 'context_tester', TEST_TOKEN)])
    monkeypatch.setenv('LIPNET_ACCESS_FILE', str(config))
    original = security.hmac.compare_digest
    calls = []

    def checked(left, right):
        calls.append(True)
        return original(left, right)

    monkeypatch.setattr(security.hmac, 'compare_digest', checked)
    assert security.access('Bearer ' + TEST_TOKEN)['actor'] == 'runtime-test'
    assert calls


def test_duplicate_principal_ids_fail_closed(monkeypatch, tmp_path):
    config = _access_file(tmp_path / 'access.json', [
        ('runtime-test', 'context_tester', TEST_TOKEN),
        ('runtime-test', 'collector', OTHER_TOKEN),
    ])
    monkeypatch.setenv('LIPNET_ACCESS_FILE', str(config))
    with pytest.raises(HTTPException) as error:
        security.access('Bearer ' + TEST_TOKEN)
    assert error.value.status_code == 503


def test_context_tester_route_matrix_is_exact():
    allowed = {
        ('POST', '/api/context/interpret'),
        ('POST', '/api/context/sessions'),
        ('POST', '/api/context/sessions/consent'),
        ('POST', '/api/context/sessions/consent/revoke'),
        ('POST', '/api/context/sessions/terminate'),
    }
    assert all(security.context_tester_route_allowed(path, method) for method, path in allowed)
    for method, path in [
        ('POST', '/api/context/feedback'),
        ('GET', '/api/context/health'),
        ('POST', '/api/adaptive/captures'),
        ('POST', '/api/recording/start'),
        ('POST', '/api/adaptive/participants/verify'),
        ('POST', '/api/adaptive/models/x/deploy'),
    ]:
        assert not security.context_tester_route_allowed(path, method)


def test_context_test_access_file_paths_are_git_ignored():
    repository = credentials_cli.Path(__file__).resolve().parents[1]
    private_directory = repository / 'data' / 'adaptive_private' / 'runtime_test_access'
    access_path, token_path = private_directory / 'access.json', private_directory / 'tokens.json'
    assert credentials_cli._credentials_are_ignored(private_directory, (access_path, token_path))
    unsafe_directory = repository / 'runtime_access_test_fixture'
    assert not credentials_cli._credentials_are_ignored(
        unsafe_directory, (unsafe_directory / 'access.json', unsafe_directory / 'tokens.json'))


def test_context_test_provisioning_is_opt_in_and_hash_only(monkeypatch, tmp_path, capsys):
    directory = tmp_path / 'runtime-access'
    monkeypatch.setattr(credentials_cli, '_windows_acl_is_private', lambda _path: True)
    monkeypatch.setattr(credentials_cli, '_credentials_are_ignored', lambda *_args: True)
    monkeypatch.setattr(credentials_cli.secrets, 'token_urlsafe', lambda _count: TEST_TOKEN)
    credentials_cli.main(['--context-test', '--directory', str(directory)])

    output = (directory / 'access.json').read_text(encoding='utf-8')
    tokens = json.loads((directory / 'tokens.json').read_text(encoding='utf-8'))
    assert TEST_TOKEN not in output
    assert json.loads(output) == {'principals': [{
        'id': 'runtime-test', 'role': 'context_tester',
        'token_sha256': [hashlib.sha256(TEST_TOKEN.encode()).hexdigest()],
    }]}
    assert tokens == {'runtime-test': TEST_TOKEN}
    assert TEST_TOKEN not in capsys.readouterr().out


def test_default_provisioning_keeps_existing_principal_set(monkeypatch, tmp_path, capsys):
    directory = tmp_path / 'legacy-access'
    tokens = iter(('fixture-operator-' + 'o' * 40, 'fixture-collector-' + 'c' * 40))
    monkeypatch.setattr(credentials_cli, '_windows_acl_is_private', lambda _path: True)
    monkeypatch.setattr(credentials_cli, '_credentials_are_ignored', lambda *_args: True)
    monkeypatch.setattr(credentials_cli.secrets, 'token_urlsafe', lambda _count: next(tokens))
    credentials_cli.main(['--directory', str(directory)])
    access_value = json.loads((directory / 'access.json').read_text(encoding='utf-8'))
    assert {(row['id'], row['role']) for row in access_value['principals']} == {
        ('local-operator', 'reviewer'), ('local-collector', 'collector')}
    assert 'fixture-operator' not in capsys.readouterr().out


def test_context_test_provisioning_refuses_existing_files_without_overwrite(monkeypatch, tmp_path):
    directory = tmp_path / 'existing'
    directory.mkdir()
    access_file, token_file = directory / 'access.json', directory / 'tokens.json'
    access_file.write_text('existing access', encoding='utf-8')
    token_file.write_text('existing token store', encoding='utf-8')
    before = (access_file.read_bytes(), token_file.read_bytes())
    monkeypatch.setattr(credentials_cli, '_windows_acl_is_private', lambda _path: True)
    monkeypatch.setattr(credentials_cli, '_credentials_are_ignored', lambda *_args: True)
    with pytest.raises(SystemExit):
        credentials_cli.main(['--context-test', '--directory', str(directory)])
    assert (access_file.read_bytes(), token_file.read_bytes()) == before


def test_unsafe_acl_fails_before_writing(monkeypatch, tmp_path):
    directory = tmp_path / 'unsafe'
    monkeypatch.setattr(credentials_cli, '_windows_acl_is_private', lambda _path: False)
    monkeypatch.setattr(credentials_cli, '_credentials_are_ignored', lambda *_args: True)
    with pytest.raises(SystemExit):
        credentials_cli.main(['--context-test', '--directory', str(directory)])
    assert not (directory / 'access.json').exists()
    assert not (directory / 'tokens.json').exists()


def test_acl_failure_after_write_removes_all_partial_credentials(monkeypatch, tmp_path):
    directory = tmp_path / 'verification-fails'
    calls = []

    def verifier(path):
        calls.append(path.name)
        return len(calls) < 4

    monkeypatch.setattr(credentials_cli, '_windows_acl_is_private', verifier)
    monkeypatch.setattr(credentials_cli, '_credentials_are_ignored', lambda *_args: True)
    monkeypatch.setattr(credentials_cli.secrets, 'token_urlsafe', lambda _count: TEST_TOKEN)
    with pytest.raises(SystemExit):
        credentials_cli.main(['--context-test', '--directory', str(directory)])
    assert calls == [directory.name, directory.name, 'access.json', 'tokens.json']
    assert not (directory / 'access.json').exists()
    assert not (directory / 'tokens.json').exists()


@pytest.fixture
def context_tester_client(monkeypatch, tmp_path):
    db_path = tmp_path / 'feedback.sqlite3'
    db = sqlite3.connect(db_path)
    db.executescript('''
      CREATE TABLE participants(owner TEXT NOT NULL,id TEXT NOT NULL,verified_by TEXT,
        permission TEXT,PRIMARY KEY(owner,id));
      CREATE TABLE captures(id TEXT PRIMARY KEY);
      PRAGMA user_version=0;
    ''')
    db.close()
    migrate(db_path)
    monkeypatch.setattr(context_routes, 'context_session_store', ContextSessionStore(db_path))
    config = _access_file(tmp_path / 'access.json')
    monkeypatch.setenv('LIPNET_ACCESS_FILE', str(config))
    monkeypatch.setenv('LIPNET_LLM_ENABLED', '0')
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    app = FastAPI()
    app.include_router(context_routes.router)
    app.add_middleware(security.RequestBoundary)
    return TestClient(app), db_path


def test_context_test_session_consent_and_termination_without_provider(
        context_tester_client, monkeypatch):
    client, db_path = context_tester_client
    import app.services.context_interpretation as interpretation

    def forbidden_provider(*_args, **_kwargs):
        raise AssertionError('provider must not be called in auth tests')

    monkeypatch.setattr(interpretation, '_post_json', forbidden_provider)
    evicted = []
    original_evict = context_routes.interpretation_service.evict_session

    def record_eviction(actor, session_id):
        evicted.append((actor, session_id))
        original_evict(actor, session_id)

    monkeypatch.setattr(context_routes.interpretation_service, 'evict_session', record_eviction)
    headers = {'Authorization': 'Bearer ' + TEST_TOKEN}
    assert client.post('/api/context/sessions', json={}, headers={}).status_code == 401
    created = client.post('/api/context/sessions', json={}, headers=headers)
    assert created.status_code == 201
    payload = created.json()
    session_token = payload['session_token']
    assert payload['participant_bound'] is False
    scoped = {**headers, 'X-Interpretation-Session': session_token}
    request = {
        'mode': 'word', 'event_id': 'runtime_test', 'visual_candidate': 'hello',
        'original_word': None, 'visual_uncertain': True,
        'candidates': [{'word': 'hello', 'prob': .55}, {'word': 'next', 'prob': .35}],
        'context': 'At a cafe menu', 'history': [], 'consent_text': True,
    }
    assert client.post('/api/context/interpret', json=request, headers=scoped).status_code == 403
    assert client.post('/api/context/interpret', json=request,
                       headers={**headers, 'X-Interpretation-Session': 'invalid-session-token'}).status_code in {401, 403}

    denied_owner = {'Authorization': 'Bearer ' + OTHER_TOKEN,
                    'X-Interpretation-Session': session_token}
    assert client.post('/api/context/sessions/consent', json={
        'purpose': PURPOSE, 'notice_version': NOTICE_VERSION, 'granted': True,
    }, headers=denied_owner).status_code in {401, 403}
    assert client.post('/api/context/sessions/terminate', headers=denied_owner).status_code in {401, 403}

    grant = client.post('/api/context/sessions/consent', json={
        'purpose': PURPOSE, 'notice_version': NOTICE_VERSION, 'granted': True,
    }, headers=scoped)
    assert grant.status_code == 200
    authorized = client.post('/api/context/interpret', json=request, headers=scoped)
    assert authorized.status_code == 200
    assert authorized.json()['status'] == 'unavailable'
    assert authorized.json()['safe_prediction'] is None
    revoke = client.post('/api/context/sessions/consent/revoke', headers=scoped)
    assert revoke.status_code == 200
    assert client.post('/api/context/interpret', json=request, headers=scoped).status_code == 403
    terminated = client.post('/api/context/sessions/terminate', headers=scoped)
    assert terminated.status_code == 200
    assert client.post('/api/context/interpret', json=request, headers=scoped).status_code == 401
    assert client.post('/api/context/sessions/terminate', headers=scoped).status_code == 401
    assert len(evicted) == 2
    assert all(actor == 'runtime-test' for actor, _ in evicted)
    assert evicted[0][1] == evicted[1][1]

    with sqlite3.connect(db_path) as db:
        row = db.execute(
            'SELECT token_sha256,owner,participant_id,terminated_at FROM context_sessions').fetchone()
    assert row[0] == hashlib.sha256(session_token.encode()).hexdigest()
    assert session_token not in row[0]
    assert row[1] == 'runtime-test' and row[2] is None and row[3] is not None
    with sqlite3.connect(db_path) as db:
        consent = db.execute(
            'SELECT purpose,revoked_at FROM context_session_consents').fetchone()
    assert consent[0] == PURPOSE and consent[1] is not None


def test_context_tester_cannot_access_unrelated_route_families(context_tester_client):
    client, _ = context_tester_client
    headers = {'Authorization': 'Bearer ' + TEST_TOKEN}
    denied = [
        ('POST', '/api/adaptive/captures'),
        ('POST', '/api/adaptive/captures/capture-1/feedback'),
        ('POST', '/api/adaptive/participants/verify'),
        ('POST', '/api/adaptive/models/model-1/deploy'),
        ('POST', '/api/adaptive/training/jobs'),
        ('POST', '/api/recording/start'),
        ('GET', '/api/recordings/A'),
        ('POST', '/api/studio/bundles'),
        ('POST', '/api/training/start'),
        ('POST', '/api/word/train-personal'),
        ('POST', '/api/context/feedback'),
    ]
    for method, path in denied:
        response = client.request(method, path, content=b'{not-json', headers=headers)
        assert response.status_code == 403, (method, path, response.status_code)
