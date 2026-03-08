FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install  torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# Artifacts must be pre-built by running scripts/01_ingest.py 
# and scripts/02_cluster.py before building the image
COPY . .

# Add /app to the pythonpath so src is natively resolvable
ENV PYTHONPATH="/app"

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
