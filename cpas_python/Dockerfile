FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8050

CMD gunicorn app:server \
    --workers 2 \
    --bind 0.0.0.0:${PORT:-8050} \
    --timeout 120
