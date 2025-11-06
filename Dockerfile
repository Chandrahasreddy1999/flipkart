# Stage 1: Build dependencies
FROM python:3.10-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final runtime image
FROM python:3.10-slim
WORKDIR /app

# Copy installed packages and source code
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY . .

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Expose app port
EXPOSE 5000

# Run FastAPI app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
