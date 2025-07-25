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

# Download original safetensors models (not GGUF)
RUN wget -O /app/ve.safetensors https://huggingface.co/ResembleAI/chatterbox/resolve/main/ve.safetensors
RUN wget -O /app/t3_cfg.safetensors https://huggingface.co/ResembleAI/chatterbox/resolve/main/t3_cfg.safetensors
RUN wget -O /app/s3gen.safetensors https://huggingface.co/ResembleAI/chatterbox/resolve/main/s3gen.safetensors
RUN wget -O /app/tokenizer.json https://huggingface.co/ResembleAI/chatterbox/resolve/main/tokenizer.json

# Clone your quantization-enabled chatterbox repo
RUN git clone https://github.com/ekkus93/chatterbox.git /tmp/chatterbox && \
    cd /tmp/chatterbox && \
    FORCE_CUDA=0 pip install --no-cache-dir . --extra-index-url https://download.pytorch.org/whl/cpu

# Create models directory
RUN mkdir -p /app/models/box

# Copy quantization script and run it
COPY quantize_models.py /app/quantize_models.py
RUN python /app/quantize_models.py

# Clean up only the processed model files (keep s3gen.safetensors for direct use)
RUN rm /app/ve.safetensors /app/t3_cfg.safetensors

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

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy quantized model files and app
COPY --from=builder /app/models /app/models
COPY --from=builder /app/s3gen.safetensors /app/s3gen.safetensors
COPY --from=builder /app/tokenizer.json /app/tokenizer.json
COPY --from=builder /app/main.py /app/main.py
COPY --from=builder /app/minah1.wav /app/minah1.wav

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]