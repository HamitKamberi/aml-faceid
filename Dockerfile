FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/app/data \
    RAW_DIR=/app/dataset/dataset/ \
    TZ=Europe/Zurich

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libglib2.0-0 libgomp1 v4l-utils tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements_docker.txt .
RUN pip install -r requirements_docker.txt

COPY faceeid.py attendance.py try.py ./

EXPOSE 5002

CMD ["python", "attendance.py"]
