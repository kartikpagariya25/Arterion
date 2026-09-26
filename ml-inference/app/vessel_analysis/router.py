import json
from typing import Optional
from fastapi import APIRouter, Header
from pydantic import BaseModel

from .model import predict
from .report_generator import generate_report
from .video_utils import predict_video
from ..auth.router import get_current_user
from ..auth.db import get_db

router = APIRouter(prefix="/vessel-analysis", tags=["vessel-analysis"])


class PredictRequest(BaseModel):
    file_url: str


class VideoPredictRequest(BaseModel):
    file_url: str
    frame_stride: int = 5


def _save_scan(user_id: int, analysis: dict, report: dict):
    with get_db() as conn:
        conn.execute(
            """INSERT INTO scans (user_id, original_image_base64, overlay_image_base64, summary, confidence, lesions_json, report_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                analysis.get("original_image_base64"),
                analysis.get("overlay_image_base64"),
                analysis.get("summary"),
                analysis.get("confidence"),
                json.dumps(analysis.get("lesions", [])),
                json.dumps(report),
            ),
        )


@router.post("/predict")
def predict_vessel(req: PredictRequest):
    return predict(req.file_url)


@router.post("/report")
def predict_with_report(req: PredictRequest, authorization: Optional[str] = Header(None)):
    analysis = predict(req.file_url)
    report = generate_report(analysis)

    user = None
    if authorization:
        try:
            user = get_current_user(authorization)
        except Exception:
            user = None

    if user:
        _save_scan(user["id"], analysis, report)
        # Patients only ever see the simplified patient-facing report.
        # Doctors get both the detailed clinical report and the simplified version.
        if user["role"] == "patient":
            report = {"patient_report": report.get("patient_report"), "disclaimer": report.get("disclaimer")}

    return {**analysis, "report": report, "viewer_role": user["role"] if user else None}


@router.post("/predict-video")
def predict_vessel_video(req: VideoPredictRequest):
    return predict_video(req.file_url, frame_stride=req.frame_stride)
