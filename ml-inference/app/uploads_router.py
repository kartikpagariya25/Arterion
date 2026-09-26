import os
import uuid
from fastapi import APIRouter, UploadFile, File, Request

router = APIRouter(tags=["uploads"])

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "static_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/uploads")
async def upload_file(request: Request, file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1] or ".png"
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)

    contents = await file.read()
    with open(filepath, "wb") as f:
        f.write(contents)

    base_url = str(request.base_url).rstrip("/")
    file_url = f"{base_url}/static_uploads/{filename}"
    return {"file_url": file_url}
