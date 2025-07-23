# ---- Step 1: Builder ----
FROM continuumio/miniconda3 AS builder

WORKDIR /app

# Copy environment and install conda env
COPY environment.yml .
RUN conda env create -f environment.yml
RUN echo "conda activate chatterbox_gguf" >> ~/.bashrc

# Install additional Python dependencies (if needed)
RUN conda run -n chatterbox_gguf pip install huggingface_hub

# Clone required repos from GitHub
RUN git clone https://github.com/resemble-ai/chatterbox.git /app/chatterbox
RUN git clone https://github.com/calcuis/gguf-connector.git /app/gguf-connector
RUN conda run -n chatterbox_gguf pip install -e /app/chatterbox

# Set PYTHONPATH so Python can find gguf_connector
ENV PYTHONPATH="/app/gguf-connector/src"

# Download GGUF model files and tokenizer.json
RUN wget -O /app/ve_fp32-f16.gguf https://huggingface.co/calcuis/chatterbox-gguf/resolve/main/ve_fp32-f16.gguf
RUN wget -O /app/s3gen-bf16.gguf https://huggingface.co/calcuis/chatterbox-gguf/resolve/main/s3gen-bf16.gguf
RUN wget -O /app/t3_cfg-q4_k_m.gguf https://huggingface.co/calcuis/chatterbox-gguf/resolve/main/t3_cfg-q4_k_m.gguf
RUN wget -O /app/tokenizer.json https://huggingface.co/ResembleAI/chatterbox/resolve/main/tokenizer.json

# Convert GGUF files to safetensors during build
RUN mkdir -p /app/models/box
RUN conda run -n chatterbox_gguf python -c "\
from gguf_connector.quant3 import convert_gguf_to_safetensors; \
convert_gguf_to_safetensors('/app/ve_fp32-f16.gguf', '/app/models/box/ve_fp32-f16-f32.safetensors', use_bf16=False); \
convert_gguf_to_safetensors('/app/t3_cfg-q4_k_m.gguf', '/app/models/box/t3_cfg-q4_k_m-f32.safetensors', use_bf16=False); \
convert_gguf_to_safetensors('/app/s3gen-bf16.gguf', '/app/models/box/s3gen-bf16-f32.safetensors', use_bf16=False)"

# Copy FastAPI app
COPY main.py /app/main.py

# ---- Step 2: Final ----
FROM continuumio/miniconda3

WORKDIR /app

# Copy conda env and all files from builder
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /app /app

ENV PATH="/opt/conda/envs/chatterbox_gguf/bin:$PATH"

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
