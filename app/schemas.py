from datetime import datetime

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    timestamp: datetime
    values: list[float] = Field(min_length=169)


class PredictionResponse(BaseModel):
    timestamp: datetime
    forecast: float