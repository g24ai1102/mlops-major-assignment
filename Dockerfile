# Use official Python image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
#COPY artifacts/ artifacts/

# Train the model first
RUN python src/train.py

# Default command
CMD ["python", "src/predict.py"]