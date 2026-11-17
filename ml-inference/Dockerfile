FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional extra deps, if present
COPY requirements-add.tx[t] .
RUN if [ -f requirements-add.txt ]; then pip install --no-cache-dir -r requirements-add.txt; fi

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
