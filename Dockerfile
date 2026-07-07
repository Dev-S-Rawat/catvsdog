FROM python:3.11-slim

# prevent python files from writing .pyc files to disk and keeps log streaming live
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# installing the bare minimum and cleaning downloaded catalogue
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# installing requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY src/ ./src/
COPY weights/ ./weights/
COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]