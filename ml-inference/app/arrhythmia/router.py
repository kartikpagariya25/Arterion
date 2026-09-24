from fastapi import APIRouter
from pydantic import BaseModel
from .model import predict

router = APIRouter(prefix="/arrhythmia", tags=["arrhythmia"])


class PredictRequest(BaseModel):
    file_url: str


@router.post("/predict")
def predict_arrhythmia(req: PredictRequest):
    return predict(req.file_url)
