FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD gunicorn app:server \
    --workers 1 \
    --threads 4 \
    --worker-class gthread \
    --bind 0.0.0.0:${PORT:-8080} \
    --timeout 300
