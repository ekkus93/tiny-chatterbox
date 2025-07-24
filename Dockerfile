# ---- Step 1: Builder ----
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables to force CPU-only installations
ENV TORCH_CUDA_ARCH_LIST=""
ENV FORCE_CUDA="0"
ENV CUDA_VISIBLE_DEVICES=""

# Install PyTorch CPU-only first with explicit CPU index
RUN pip install --no-cache-dir \
    torch==2.5.1+cpu \
    torchaudio==2.5.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu \
    --no-deps

# Copy requirements and install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Clone and install chatterbox - create a wheel for later use
RUN git clone https://github.com/resemble-ai/chatterbox.git /tmp/chatterbox && \
    cd /tmp/chatterbox && \
    FORCE_CUDA=0 pip wheel --no-cache-dir . --wheel-dir /app/wheels --extra-index-url https://download.pytorch.org/whl/cpu

# Clone gguf-connector (only the needed parts)
RUN git clone https://github.com/calcuis/gguf-connector.git /tmp/gguf-connector

# Download only necessary GGUF model files and tokenizer
RUN wget -O /app/ve_fp32-f16.gguf https://huggingface.co/calcuis/chatterbox-gguf/resolve/main/ve_fp32-f16.gguf
RUN wget -O /app/t3_cfg-q4_k_m.gguf https://huggingface.co/calcuis/chatterbox-gguf/resolve/main/t3_cfg-q4_k_m.gguf
RUN wget -O /app/s3gen-bf16.gguf https://huggingface.co/calcuis/chatterbox-gguf/resolve/main/s3gen-bf16.gguf
RUN wget -O /app/tokenizer.json https://huggingface.co/ResembleAI/chatterbox/resolve/main/tokenizer.json

# Convert GGUF files to safetensors
RUN mkdir -p /app/models/box
RUN PYTHONPATH="/tmp/gguf-connector/src" python -c "\
from gguf_connector.quant3 import convert_gguf_to_safetensors; \
convert_gguf_to_safetensors('/app/ve_fp32-f16.gguf', '/app/models/box/ve_fp32-f16-f32.safetensors', use_bf16=False); \
convert_gguf_to_safetensors('/app/t3_cfg-q4_k_m.gguf', '/app/models/box/t3_cfg-q4_k_m-f32.safetensors', use_bf16=False); \
convert_gguf_to_safetensors('/app/s3gen-bf16.gguf', '/app/models/box/s3gen-bf16-f32.safetensors', use_bf16=False)"

# Clean up GGUF files after conversion
RUN rm /app/*.gguf

# Copy FastAPI app and reference audio
COPY main.py /app/main.py
COPY minah1.wav /app/minah1.wav

# ---- Final stage ----
FROM python:3.10-slim

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set CPU-only environment variables for runtime
ENV TORCH_CUDA_ARCH_LIST=""
ENV FORCE_CUDA="0"
ENV CUDA_VISIBLE_DEVICES=""

# Install only runtime Python dependencies (much smaller)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch==2.5.1+cpu \
    torchaudio==2.5.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu \
    --no-deps && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Install chatterbox from pre-built wheel
COPY --from=builder /app/wheels /tmp/wheels
RUN pip install --no-cache-dir /tmp/wheels/*.whl && rm -rf /tmp/wheels

# Copy only the essential files from builder
COPY --from=builder /app/models /app/models
COPY --from=builder /app/tokenizer.json /app/tokenizer.json
COPY --from=builder /app/main.py /app/main.py
COPY --from=builder /app/minah1.wav /app/minah1.wav

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]