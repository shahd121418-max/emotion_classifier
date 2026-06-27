from contextlib import asynccontextmanager

from fastapi import FastAPI

from predictor import EmotionPredictor
from schemas import (
    PredictionRequest,
    PredictionResponse,
)

predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):

    global predictor

    predictor = EmotionPredictor()

    yield


app = FastAPI(
    title="Emotion Classifier",
    lifespan=lifespan,
)


@app.get("/")
def root():

    return {
        "status": "running"
    }




@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
):

    result = predictor.predict(
        request.text
    )

    return result
