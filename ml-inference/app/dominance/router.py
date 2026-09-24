from fastapi import APIRouter
from pydantic import BaseModel
from .model import predict

router = APIRouter(prefix="/dominance", tags=["dominance"])


class PredictRequest(BaseModel):
    file_url: str


@router.post("/predict")
def predict_dominance(req: PredictRequest):
    return predict(req.file_url)
