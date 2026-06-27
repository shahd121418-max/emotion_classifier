from pydantic import BaseModel


class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    emotion: str
    confidence: float