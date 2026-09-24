from fastapi import APIRouter
from pydantic import BaseModel
from .model import predict

router = APIRouter(prefix="/lv-outline", tags=["lv-outline"])


class PredictRequest(BaseModel):
    file_url: str


@router.post("/predict")
def predict_lv(req: PredictRequest):
    return predict(req.file_url)
