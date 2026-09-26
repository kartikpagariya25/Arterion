import json
from typing import Optional
from fastapi import APIRouter, Header, HTTPException

from .router import get_current_user
from .db import get_db

router = APIRouter(prefix="/scans", tags=["scans"])


@router.get("/mine")
def my_scans(authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)

    with get_db() as conn:
        rows = conn.execute(
            """SELECT id, overlay_image_base64, summary, lesions_json, report_json, created_at
               FROM scans WHERE user_id = ? ORDER BY created_at DESC""",
            (user["id"],),
        ).fetchall()

    return [
        {
            "id": r["id"],
            "overlay_image_base64": r["overlay_image_base64"],
            "summary": r["summary"],
            "lesions": json.loads(r["lesions_json"] or "[]"),
            "report": json.loads(r["report_json"] or "null"),
            "created_at": r["created_at"],
        }
        for r in rows
    ]


@router.get("/all-patients")
def all_patients(authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    if user["role"] != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can view the patient list")

    with get_db() as conn:
        patients = conn.execute(
            "SELECT id, full_name, email, age, gender, phone FROM users WHERE role = 'patient' ORDER BY full_name"
        ).fetchall()

        result = []
        for p in patients:
            scan_rows = conn.execute(
                "SELECT summary, lesions_json, created_at FROM scans WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
                (p["id"],),
            ).fetchall()
            scan_count = conn.execute(
                "SELECT COUNT(*) as c FROM scans WHERE user_id = ?", (p["id"],)
            ).fetchone()["c"]

            latest = scan_rows[0] if scan_rows else None
            latest_severity = None
            if latest:
                lesions = json.loads(latest["lesions_json"] or "[]")
                if lesions:
                    latest_severity = max(lesions, key=lambda l: l["stenosis_percent"])["severity_band"]

            result.append({
                "id": p["id"],
                "full_name": p["full_name"],
                "email": p["email"],
                "age": p["age"],
                "gender": p["gender"],
                "phone": p["phone"],
                "scan_count": scan_count,
                "latest_scan_at": latest["created_at"] if latest else None,
                "latest_severity": latest_severity,
            })

    return result