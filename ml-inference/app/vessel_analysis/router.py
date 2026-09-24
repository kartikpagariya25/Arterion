from fastapi import APIRouter
from pydantic import BaseModel
from .model import predict

router = APIRouter(prefix="/vessel-analysis", tags=["vessel-analysis"])


class PredictRequest(BaseModel):
    file_url: str


@router.post("/predict")
def predict_vessel(req: PredictRequest):
    return predict(req.file_url)
