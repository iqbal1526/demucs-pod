FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install your Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your server code
COPY pod_server.py .

# Expose HTTP port
EXPOSE 8000

# Run Uvicorn on GPU
CMD ["uvicorn", "pod_server:app", "--host", "0.0.0.0", "--port", "8000"]
