# Ambiente reprodutível. Uso:
#   docker build -t respira-sp .
#   docker run --rm -v "$PWD/models:/app/models" -v "$PWD/reports:/app/reports" respira-sp make all-sample
# Não monte uma pasta data vazia em /app/data: ela esconderia a amostra incluída na imagem.
FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git make curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-llm.txt requirements-nn.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# llama.cpp genérico para x86-64 (sem -march=native, para rodar em qualquer CPU com AVX2)
RUN CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON" \
        CMAKE_BUILD_PARALLEL_LEVEL=2 pip install --no-cache-dir -r requirements-llm.txt && \
        pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu && \
        pip install --no-cache-dir -r requirements-nn.txt
COPY . .
ENV PYTHONPATH=/app PY=python
CMD ["make", "help"]
