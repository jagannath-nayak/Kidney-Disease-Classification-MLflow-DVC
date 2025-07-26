FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies TensorFlow needs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements first
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy the entire app
COPY . .

# Set PORT for Render
ENV PORT=8080

# Start app with gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
