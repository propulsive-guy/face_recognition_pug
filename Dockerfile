# ------------------- Base Image -------------------
FROM python:3.11-slim

# ------------------- Set Environment -------------------
# Prevent Python from writing pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ------------------- Set Working Directory -------------------
WORKDIR /app

# ------------------- Install System Dependencies -------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ------------------- Copy Requirements -------------------
COPY requirements.txt .

# ------------------- Install Python Dependencies -------------------
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ------------------- Copy App Code -------------------
COPY . .

# ------------------- Expose Port -------------------
# Cloud Run uses PORT environment variable automatically
EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]

