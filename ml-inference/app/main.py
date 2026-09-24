from fastapi import FastAPI
from app.vessel_analysis.router import router as vessel_router
from app.ef_estimation.router import router as ef_router
from app.lv_outline.router import router as lv_router
from app.dominance.router import router as dominance_router
from app.arrhythmia.router import router as arrhythmia_router

app = FastAPI(title="CardioSense ML Inference Service")

app.include_router(vessel_router)
app.include_router(ef_router)
app.include_router(lv_router)
app.include_router(dominance_router)
app.include_router(arrhythmia_router)


@app.get("/health")
def health():
    return {"status": "ok"}
