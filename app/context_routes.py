"""Separate optional post-processing endpoint; visual inference is unchanged."""
import logging
from typing import Literal, Annotated
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from app.security import access, CONTEXT_TESTER_ROLE, context_tester_route_allowed
from app.services.context_interpretation import InterpretationRequest,configuration,interpretation_service,vocabulary
from app.services.context_sessions import (ContextSessionError, context_session_store,
                                           PURPOSE, NOTICE_VERSION)

router = APIRouter()


def context_session_user(request: Request, user=Depends(access)):
    """Session/interpreter routes accept existing identities plus the scoped test role."""
    if user.get('role') == CONTEXT_TESTER_ROLE and not context_tester_route_allowed(
            request.url.path, request.method):
        raise HTTPException(403, 'Context test credentials are limited to interpretation sessions.')
    if user.get('role') not in {'collector', 'reviewer', CONTEXT_TESTER_ROLE}:
        raise HTTPException(403, 'This identity cannot use context interpretation.')
    return user


def context_feedback_user(user=Depends(access)):
    """The test identity cannot write context feedback outside session lifecycle."""
    if user.get('role') == CONTEXT_TESTER_ROLE:
        raise HTTPException(403, 'Context test credentials cannot submit interpretation feedback.')
    return user


@router.get('/api/context/health')
def context_health():
    return configuration()


@router.post('/api/context/interpret')
async def interpret(request:InterpretationRequest,user=Depends(context_session_user),
                    session_token:Annotated[str|None,Header(alias='X-Interpretation-Session')]=None):
    if not session_token:
        raise HTTPException(401,'An active interpretation session is required.')
    try:
        scope=context_session_store.authorize(user['actor'],session_token,require_consent=True)
        result=await interpretation_service.interpret(request,user['actor'],scope,scope['consent_id'],context_session_store)
        if not context_session_store.is_current(user['actor'],scope,scope['consent_id']):
            interpretation_service.evict_session(user['actor'],scope['session_id'])
            result.update(suggested_word=None,suggested_visual_score=None,llm_suggestion=None,
                          requires_confirmation=False,status='session_inactive',cached=False)
        return result
    except ContextSessionError as error:
        raise HTTPException(error.status,error.code) from None


class CreateContextSession(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    participant_id: str|None=Field(default=None,min_length=1,max_length=80,pattern=r'^[A-Za-z0-9_-]+$')


class ContextConsent(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    purpose: Literal['text_only_llm_interpretation']
    notice_version: str=Field(min_length=1,max_length=40)
    granted: bool


def _require_session(session_token):
    if not session_token:
        raise HTTPException(401,'An active interpretation session is required.')
    return session_token


@router.post('/api/context/sessions',status_code=201)
def create_context_session(request:CreateContextSession,user=Depends(context_session_user)):
    try:
        return context_session_store.create(user['actor'],request.participant_id)
    except ContextSessionError as error:
        raise HTTPException(error.status,error.code) from None


@router.post('/api/context/sessions/consent')
def grant_context_consent(request:ContextConsent,user=Depends(context_session_user),
                          session_token:Annotated[str|None,Header(alias='X-Interpretation-Session')]=None):
    try:
        scope=context_session_store.grant_consent(user['actor'],_require_session(session_token),
                                                  request.purpose,request.notice_version,request.granted)
        return {'consented':True,'purpose':PURPOSE,'notice_version':NOTICE_VERSION}
    except ContextSessionError as error:
        raise HTTPException(error.status,error.code) from None


@router.post('/api/context/sessions/consent/revoke')
def revoke_context_consent(user=Depends(context_session_user),
                           session_token:Annotated[str|None,Header(alias='X-Interpretation-Session')]=None):
    try:
        scope=context_session_store.revoke_consent(user['actor'],_require_session(session_token))
        interpretation_service.evict_session(user['actor'],scope['session_id'])
        return {'consented':False,'revoked':True}
    except ContextSessionError as error:
        raise HTTPException(error.status,error.code) from None


@router.post('/api/context/sessions/terminate')
def terminate_context_session(user=Depends(context_session_user),
                              session_token:Annotated[str|None,Header(alias='X-Interpretation-Session')]=None):
    try:
        scope=context_session_store.terminate(user['actor'],_require_session(session_token))
        interpretation_service.evict_session(user['actor'],scope['session_id'])
        return {'terminated':True}
    except ContextSessionError as error:
        raise HTTPException(error.status,error.code) from None


class InterpretationFeedback(BaseModel):
    model_config = ConfigDict(extra='forbid',strict=True)
    mode: Literal['word','expanded']
    event_id: str = Field(min_length=1,max_length=80,pattern=r'^[A-Za-z0-9_-]+$')
    word: str = Field(min_length=1,max_length=40,pattern=r'^[A-Za-z]+$')
    accepted: bool


@router.post('/api/context/feedback')
async def feedback(request:InterpretationFeedback,user=Depends(context_feedback_user)):
    if request.word not in vocabulary(request.mode):
        raise HTTPException(422,'Unknown vocabulary word')
    # Human-reported choice is not a verified label or a training example.
    logging.getLogger(__name__).info('llm_user_reported_interpretation mode=%s accepted=%s actor=%s',request.mode,request.accepted,user['actor'])
    return {'recorded':True,'visual_prediction_changed':False,'used_for_training':False}
