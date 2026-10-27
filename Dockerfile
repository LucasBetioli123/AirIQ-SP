# Ambiente reprodutível. Uso:
#   docker build -t teste-ia .
#   docker run --rm -it -v $PWD/data:/app/data -v $PWD/models:/app/models -p 8000:8000 teste-ia make all-sample
FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git make curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-llm.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# llama.cpp genérico para x86-64 (sem -march=native, para rodar em qualquer CPU com AVX2)
RUN CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON" \
        pip install --no-cache-dir -r requirements-llm.txt && pip install --no-cache-dir -r requirements-nn.txt
COPY . .
ENV PYTHONPATH=/app PY=python
CMD ["make", "help"]
