FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir sentence-transformers==2.7.0 numpy==1.26.4

COPY . .

CMD ["python", "rank.py", \
     "--candidates", "./candidates.jsonl.gz", \
     "--embeddings", "./embeddings.npy", \
     "--ids", "./candidate_ids.npy", \
     "--jd", "./jd_embedding.npy", \
     "--out", "./submission.csv"]
