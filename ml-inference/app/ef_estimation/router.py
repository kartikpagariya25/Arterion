from fastapi import APIRouter
from pydantic import BaseModel
from .model import predict

router = APIRouter(prefix="/ef-estimation", tags=["ef-estimation"])


class PredictRequest(BaseModel):
    file_url: str


@router.post("/predict")
def predict_ef(req: PredictRequest):
    return predict(req.file_url)
