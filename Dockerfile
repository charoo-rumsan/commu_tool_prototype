# 🐍 Use slim Python base
FROM python:3.11-slim

WORKDIR /app

# 🧩 Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 🔽 Copy requirements early to leverage Docker cache
COPY requirements.txt .

# ⚙️ Install dependencies in a single layer with retry + optimized index
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt --timeout 300 --retries 5

# 📁 Copy app files
COPY app ./app

# 🏃 Run FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
