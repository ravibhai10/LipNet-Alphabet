"""Validated resource and private-retention settings for the supported profile."""
from functools import lru_cache
import os
from pathlib import Path
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseModel):
    retention_days: int = Field(default=90, ge=1, le=3650)
    maintenance_seconds: int = Field(default=60, ge=1, le=3600)
    private_quota_bytes: int = Field(default=2 * 1024**3, ge=1024**2)
    inference_timeout_seconds: float = Field(default=30, ge=.01, le=300)
    cpu_threads: int = Field(default=1, ge=1, le=32)
    private_root: Path = ROOT / 'data/adaptive_private'
    server_camera_enabled: bool = False


    @field_validator('private_root')
    @classmethod
    def private_location(cls, value):
        value = (ROOT / value).resolve()
        data = (ROOT / 'data').resolve()
        if value == data or not value.is_relative_to(data):
            raise ValueError('Private storage must be a dedicated directory inside the repository data directory.')
        if any(value.is_relative_to(data / name) for name in ('external', 'grid_raw', 'lrw', 'processed')):
            raise ValueError('Private storage must not overlap public corpus directories.')
        return value


@lru_cache
def settings():
    mapping = {'retention_days': 'RETENTION_DAYS', 'maintenance_seconds': 'MAINTENANCE_SECONDS',
               'private_quota_bytes': 'PRIVATE_QUOTA_BYTES', 'inference_timeout_seconds': 'INFERENCE_TIMEOUT_SECONDS',
               'cpu_threads': 'CPU_THREADS', 'private_root': 'PRIVATE_ROOT',
               'server_camera_enabled': 'ENABLE_SERVER_CAMERA'}
    return Settings(**{k: os.environ['LIPNET_' + v] for k, v in mapping.items() if 'LIPNET_' + v in os.environ})
