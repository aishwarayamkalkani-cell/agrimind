# AgriMind AI - single container: FastAPI backend serving the pre-built React frontend.
# Works on Hugging Face Spaces (Docker SDK, port 7860) and any Docker host.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface \
    PORT=7860

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# CPU-only PyTorch keeps the image small; then the rest of the backend packages.
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && pip install -r backend/requirements.txt

COPY --chown=user:user . .
USER user

ENV AGRIMIND_STORE=/home/user/app/backend/storage

# Retrain the small crop model for this scikit-learn version and pre-download the
# multilingual embedding model so the first start is fast (both non-fatal).
RUN cd backend && python training/train_crop.py \
 && (python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')" || true)

EXPOSE 7860
WORKDIR /home/user/app/backend
CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
