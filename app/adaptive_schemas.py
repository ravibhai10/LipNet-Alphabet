"""Stable private prediction response shared with frontend integration."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class RankedLabel(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    label: str
    score: float = Field(ge=0, le=1)


class AdaptivePrediction(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    task: Literal['isolated_word', 'alphabet_recognition']
    prediction: str | None
    candidate: str
    confidence: float = Field(ge=0, le=1)
    top_k: list[RankedLabel]
    model_id: str = Field(min_length=1, max_length=128)
    model_version: str = Field(min_length=1, max_length=128)
    vocabulary: list[str] = Field(min_length=1, max_length=500)
    status: Literal['PREDICTED', 'UNCERTAIN']
    latency_ms: float | None = Field(default=None, ge=0)
    confidence_calibrated: Literal[False] = False

    @model_validator(mode='after')
    def validate_abstention(self):
        if (self.prediction is None) != (self.status == 'UNCERTAIN'):
            raise ValueError('Acceptance status and prediction disagree.')
        if self.candidate not in self.vocabulary or (self.prediction is not None and self.prediction not in self.vocabulary):
            raise ValueError('Prediction is outside the selected model vocabulary.')
        return self
