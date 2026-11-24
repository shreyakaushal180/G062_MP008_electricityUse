FROM python:3.12-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
COPY models/ ./models/
COPY params.yaml .

EXPOSE 8000

CMD ["uvicorn", "src.mp008.api:app", "--host", "0.0.0.0", "--port", "8000"]
