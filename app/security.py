"""Persistent, fail-closed local credentials and pre-parser request limits.

Credentials are configured, never generated at server startup. The JSON access
file maps stable principal IDs to roles and SHA-256 token digests. Rotation
changes a digest, not the principal or ownership. Legacy environment keys are
operator-only compatibility credentials, not identities for new private data.
"""
from __future__ import annotations
import hashlib
import hmac
import json
import os
import re
from pathlib import Path
from typing import Annotated
from fastapi import Header, HTTPException

IDENTIFIER = re.compile(r'^[A-Za-z0-9_-]{1,80}$')


def access(authorization: Annotated[str | None, Header()] = None):
    entries = []
    location = os.environ.get('LIPNET_ACCESS_FILE')
    if location:
        try:
            value = json.loads(Path(location).read_text(encoding='utf-8'))
            for principal in value['principals']:
                if not IDENTIFIER.fullmatch(principal['id']) or principal['role'] not in {'collector', 'reviewer'}:
                    raise ValueError('Invalid principal.')
                for digest in principal['token_sha256']:
                    if not re.fullmatch(r'[0-9a-f]{64}', digest):
                        raise ValueError('Invalid credential digest.')
                    entries.append((digest, principal['id'], principal['role']))
            if len({digest for digest, _, _ in entries}) != len(entries):
                raise ValueError('Ambiguous credential.')
        except (OSError, ValueError, KeyError, TypeError):
            raise HTTPException(503, 'Access configuration is unavailable or invalid.') from None
    else:
        for key, role, actor in [('LIPNET_REVIEW_TOKEN', 'reviewer', 'local-operator'),
                                 ('LIPNET_FEEDBACK_TOKEN', 'collector', 'local-collector')]:
            token = os.environ.get(key, '')
            if len(token) >= 32:
                entries.append((hashlib.sha256(token.encode()).hexdigest(), actor, role))
        if len({digest for digest, _, _ in entries}) != len(entries):
            raise HTTPException(503, 'Collector and reviewer credentials must differ.')
    if not entries:
        raise HTTPException(503, 'Sensitive operations disabled: configure persistent private credentials.')
    token = authorization[7:] if authorization and authorization.startswith('Bearer ') else ''
    candidate = hashlib.sha256(token.encode()).hexdigest()
    for expected, actor, role in entries:
        if len(token) >= 32 and hmac.compare_digest(candidate, expected):
            return {'actor': actor, 'role': role}
    raise HTTPException(401, 'A private access credential is required.', headers={'WWW-Authenticate': 'Bearer'})


def reviewer(user):
    if user['role'] != 'reviewer':
        raise HTTPException(403, 'Reviewer access is required.')
    return user


class RequestBoundary:
    """Authorize sensitive routes and bound streaming bodies before JSON parsing."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['method'] == 'OPTIONS':
            return await self.app(scope, receive, send)
        path, method = scope.get('path', ''), scope['method']
        sensitive = (path.startswith(('/api/studio', '/api/adaptive', '/api/live/')) or
                     path.startswith('/api/training/') or
                     (method != 'GET' and path.startswith('/api/dataset/')) or
                     path.startswith(('/api/recording/', '/api/recordings/')))
        headers = dict(scope.get('headers', []))
        async def reject(status, detail):
            await send({'type': 'http.response.start', 'status': status,
                        'headers': [(b'content-type', b'application/json'), (b'cache-control', b'no-store')]})
            await send({'type': 'http.response.body', 'body': json.dumps({'detail': detail}).encode()})
        if sensitive:
            try:
                user = access(headers.get(b'authorization', b'').decode('latin1') or None)
                if not path.startswith('/api/adaptive'):
                    reviewer(user)  # Legacy stores have no reliable ownership: operator only.
            except HTTPException as error:
                return await reject(error.status_code, error.detail)
        if method not in {'POST', 'PUT', 'PATCH'}:
            return await self.app(scope, receive, send)
        limit = 90 * 1024 * 1024 if path.startswith('/api/adaptive/captures') else 16 * 1024 * 1024
        messages, size = [], 0
        import asyncio
        try:
            while True:
                message = await asyncio.wait_for(receive(), 30)
                if message['type'] == 'http.disconnect':
                    return
                size += len(message.get('body', b''))
                if size > limit:
                    return await reject(413, 'Request exceeds this route\'s bounded upload limit.')
                messages.append(message)
                if not message.get('more_body'):
                    break
        except asyncio.TimeoutError:
            return await reject(408, 'Upload stalled.')
        async def replay():
            return messages.pop(0) if messages else await receive()
        await self.app(scope, replay, send)
