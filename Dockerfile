FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
COPY examples ./examples
COPY evals ./evals

ENV PYTHONPATH=/app/src
EXPOSE 7860

CMD ["python", "-m", "interviewloop.app"]
