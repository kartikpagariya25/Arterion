from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.vessel_analysis.router import router as vessel_router
from app.ef_estimation.router import router as ef_router
from app.lv_outline.router import router as lv_router
from app.dominance.router import router as dominance_router
from app.arrhythmia.router import router as arrhythmia_router
from app.uploads_router import router as uploads_router, UPLOAD_DIR
from app.auth.router import router as auth_router
from app.auth.scans_router import router as scans_router

app = FastAPI(title="CardioSense ML Inference Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(scans_router)
app.include_router(vessel_router)
app.include_router(ef_router)
app.include_router(lv_router)
app.include_router(dominance_router)
app.include_router(arrhythmia_router)
app.include_router(uploads_router)

app.mount("/static_uploads", StaticFiles(directory=UPLOAD_DIR), name="static_uploads")


@app.get("/health")
def health():
    return {"status": "ok"}
