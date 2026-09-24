# CardioSense — AI That Understands the Heart

Unified cardiac-care platform for the Synapse Hackathon (Symbiosis Pune, AI in Healthcare track).

## Structure
- `frontend/` — React + Vite + Tailwind app (patient and doctor portals)
- `backend/` — Node/Express API (auth-aware routes, talks to Supabase and ml-inference)
- `ml-inference/` — Single FastAPI service, one router per problem statement (`vessel_analysis`, `ef_estimation`, `lv_outline`, `dominance`, `arrhythmia`)
- `database/` — Supabase Postgres schema and RLS policies

## Setup
1. Create a Supabase project, run `database/schema.sql` in the SQL editor.
2. Copy `.env.example` to `.env` in both `frontend/` and `backend/`, fill in Supabase keys.
3. `cd backend && npm install && npm run dev`
4. `cd frontend && npm install && npm run dev`
5. `cd ml-inference && pip install -r requirements.txt && uvicorn app.main:app --reload`

Trained model weights for each problem statement live outside this repo during training (Colab/local GPU) and only the final checkpoint is dropped into `ml-inference/app/<problem_statement>/weights/` once ready.
