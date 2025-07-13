FROM python:3.11-slim-bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV ONNX_PROVIDERS="CPUExecutionProvider,CUDAExecutionProvider"
ENV OMP_NUM_THREADS=4
ENV ONNXRUNTIME_DEVICE_ID=0

# Expose and run the application
EXPOSE 5000
CMD ["python", "app.py"]