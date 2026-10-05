FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml README.md ./
COPY app app
COPY crawler crawler
COPY web web
COPY sources.yaml ./
# Editable install so web/, sources.yaml and mock data resolve from /app.
RUN pip install --no-cache-dir -e .
# Self-hosted models for MOCK_MODE=false: docker compose build --build-arg INSTALL_MODELS=true
ARG INSTALL_MODELS=false
RUN if [ "$INSTALL_MODELS" = "true" ]; then \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch && \
      pip install --no-cache-dir -e '.[models]'; \
    fi
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
