# Reproducible environment. Usage:
#   docker build -t airiq-sp .
#   docker run --rm -v "$PWD/models:/app/models" -v "$PWD/reports:/app/reports" airiq-sp make all-sample
# Do not mount an empty data folder on /app/data: it would hide the sample shipped in the image.
FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git make curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-llm.txt requirements-nn.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# generic x86-64 llama.cpp build (no -march=native, so it runs on any AVX2 CPU)
RUN CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON" \
        CMAKE_BUILD_PARALLEL_LEVEL=2 pip install --no-cache-dir -r requirements-llm.txt && \
        pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu && \
        pip install --no-cache-dir -r requirements-nn.txt
COPY . .
ENV PYTHONPATH=/app PY=python
CMD ["make", "help"]
