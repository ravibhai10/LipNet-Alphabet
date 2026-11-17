"""Data collection routes never fit weights or change model selection."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel,Field
from app.services.recording_studio import studio,configuration

router=APIRouter(prefix='/api/studio',tags=['recording studio'])


class StudioBundle(BaseModel):
    clips:list[dict]=Field(min_length=1,max_length=32)
    configuration_sha256:str=Field(min_length=64,max_length=64)


@router.get('/config')
def studio_config():return configuration(studio.root)


@router.get('/inventory')
def studio_inventory():return studio.inventory()


@router.post('/bundles',status_code=201)
def save_studio_bundle(request:StudioBundle):
    raise HTTPException(410, 'JPEG-only studio submission is archived. Use /api/adaptive/captures with original source video, consent and owner identity.')
    try:return studio.save(request.clips,request.configuration_sha256)
    except (ValueError,TypeError,KeyError) as error:raise HTTPException(422,str(error)) from error


@router.get('/export/{task}')
def export_studio_task(task:str):
    try:return studio.export(task)
    except ValueError as error:raise HTTPException(422,str(error)) from error
