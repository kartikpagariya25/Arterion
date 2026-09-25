# CardioSense — AI That Understands the Heart

Cardiac AI models for the Synapse Hackathon (Symbiosis Pune, AI in Healthcare track).

**Current focus: pure ML — training, evaluation, and exporting models for all 5 problem statements.** The platform layer (frontend, backend, auth, EHR database) is deliberately deferred to the hackathon itself and is not part of this repo right now.

## Structure

- `ml-training/<problem-statement>/` — one self-contained folder per PS, each with its own `PRD.md`, `IMPLEMENTATION.md`, `src/`, `data/` (gitignored), and `artifacts/` (trained weights + metrics, gitignored except the final ONNX export and `metrics.json`)
- `ml-inference/` — single deployable FastAPI service, one router per PS (`app/<problem_statement>/`), each loading its trained ONNX model from a local `weights/` folder

## Problem Statements

| Folder | Status |
|---|---|
| `ml-training/arrhythmia/` | ✅ Fully trained, evaluated, exported, integrated |
| `ml-training/vessel_analysis/` | 🔄 In progress |
| `ml-training/ef_estimation/` | Not started |
| `ml-training/lv_outline/` | Not started |
| `ml-training/dominance/` | Not started |

## Setup (per problem statement)

```bash
cd ml-training/<problem-statement>
python -m venv venv
venv\Scripts\activate
# install torch/torchvision separately, matching your exact GPU — see that PS's IMPLEMENTATION.md
pip install -r requirements.txt
```

Follow that folder's `IMPLEMENTATION.md` phase by phase.

## Running the inference service

```bash
cd ml-inference
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Each PS's trained model is picked up automatically once its `model.onnx` + supporting files are copied into `ml-inference/app/<problem_statement>/weights/`.

---

*Platform layer (auth, EHR, patient/doctor dashboards, case pipeline) is designed and scaffolded separately — it will be brought back into this repo when hackathon build time starts.*
