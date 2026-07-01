FROM python:3.13-slim

RUN apt-get update -qq && apt-get install -y -qq curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x docker-entrypoint.sh

RUN chmod +x docker-entrypoint.sh

EXPOSE ${PORT:-8000}

CMD ["./docker-entrypoint.sh"]
