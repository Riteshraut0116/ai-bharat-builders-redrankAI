FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --timeout=300 --retries=5 \
    torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir sentence-transformers==2.7.0 numpy==1.26.4

COPY . .

# Data files are expected at /data (mount at runtime)
# Output will be written to /out/submission.csv
CMD ["python", "rank.py", \
     "--candidates", "/data/candidates.jsonl", \
     "--embeddings", "/data/embeddings.npy", \
     "--ids", "/data/candidate_ids.npy", \
     "--jd", "/data/jd_embedding.npy", \
     "--out", "/out/submission.csv"]
