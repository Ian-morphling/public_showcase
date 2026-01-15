# Dockerfile

# 1. Base image
FROM python:3.10-slim

# 2. Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set working directory inside container
WORKDIR /app

# 4. Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy dependency definitions first (better caching)
COPY requirements.txt .

# 6. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy application code and required assets
COPY backend/ backend/
COPY langgraph_nodes.py .
COPY index/ index/
COPY outputs/ outputs/

# 8. Expose FastAPI port
EXPOSE 8000

# 9. Run FastAPI using uvicorn
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]