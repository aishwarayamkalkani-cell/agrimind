"""Hugging Face Spaces launcher for AgriMind AI.

The Space runs `python app.py`; this starts the FastAPI backend (which also serves
the built React frontend) on port 7860. Everything runs on CPU.
"""
import os
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only; no GPU calls outside ZeroGPU functions

BACKEND = Path(__file__).resolve().parent / "backend"
os.environ.setdefault("AGRIMIND_STORE", str(BACKEND / "storage"))
os.chdir(BACKEND)
sys.path.insert(0, str(BACKEND))

# Retrain the small crop model so it matches this server's scikit-learn version (a few seconds).
try:
    subprocess.run([sys.executable, "training/train_crop.py"], check=True, timeout=300)
except Exception as e:  # the bundled model is used if retraining fails
    print("Crop model retrain skipped:", e, flush=True)

import uvicorn  # noqa: E402

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
