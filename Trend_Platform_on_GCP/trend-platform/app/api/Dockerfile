# `-slim` uses a smaller base image than full Debian variants to reduce image size.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
EXPOSE 8080
# Starts FastAPI via Uvicorn on all interfaces so Kubernetes Service can reach it on port 8080.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
