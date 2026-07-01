FROM python:3.13-slim

RUN apt-get update -qq && apt-get install -y -qq curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download ML models at build time (saves ~3min cold start)
RUN python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('jinaai/jina-embeddings-v2-small-en', trust_remote_code=True)
from sentence_transformers import CrossEncoder
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
"

COPY . .

EXPOSE 8000

CMD ["./docker-entrypoint.sh"]
