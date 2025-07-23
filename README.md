# Tiny Chatterbox TTS Docker API

A lightweight Docker container for text-to-speech synthesis using quantized GGUF models via Chatterbox TTS.

## Features

- **Optimized Size**: Uses slim Python base image instead of conda for minimal footprint
- **Quantized Models**: Uses q4_k_m quantized models for efficient CPU inference
- **FastAPI Backend**: RESTful API with automatic documentation
- **Docker Container**: Easy deployment and consistent environment
- **Voice Cloning**: Reference audio support for voice characteristics
- **GGUF Format**: Optimized model format for faster loading and inference

## Quick Start

### 1. Build the Docker Container

```bash
cd /home/phil/work/tiny-chatterbox
docker build -t tiny-chatterbox .
```

**Build Stats:**
- Build time: ~4.5 minutes (including model downloads)
- Final image size: Significantly smaller than conda-based alternatives
- CPU-only optimized PyTorch (no CUDA dependencies)

**Note**: The optimized build process will:
- Use Python 3.10 slim base image for smaller size
- Download only necessary GGUF models (ve_fp32-f16.gguf, t3_cfg-q4_k_m.gguf, s3gen-bf16.gguf)
- Convert GGUF files to safetensors format during build
- Remove original GGUF files after conversion to save space
- Install only runtime dependencies in final image

### 2. Run the Docker Container

```bash
docker run -p 8000:8000 tiny-chatterbox
```

The API will be available at `http://localhost:8000`

**Optional**: Run with custom port mapping:
```bash
docker run -p 9000:8000 tiny-chatterbox
```

### 3. Test the API

Now you can test your TTS API with:

```bash
curl -X POST "http://localhost:8000/speak" \
  -F "text=Hello world, this is a test of the optimized Chatterbox TTS system." \
  -F "voice_sample=@/path/to/reference_audio.wav" \
  --output generated_speech.wav
```

### 4. API Documentation

Once running, visit these URLs:
- **Interactive API docs**: http://localhost:8000/docs
- **ReDoc documentation**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Container Size Optimization

This version uses several optimizations to reduce container size:

- **Slim base image**: `python:3.10-slim` instead of `continuumio/miniconda3`
- **Multi-stage build**: Separate build and runtime stages
- **CPU-only PyTorch**: Eliminates ~3-4GB of CUDA dependencies
- **Minimal dependencies**: Only essential packages via pip
- **Model cleanup**: Remove original GGUF files after conversion
- **Runtime libraries**: Only install libsndfile1 for audio processing

Expected size reduction: ~70% smaller than conda-based version

## API Usage Examples

### Using curl

#### Basic TTS with reference audio:

```bash
curl -X POST "http://localhost:8000/speak" \
  -F "text=Hello world, this is a test of the text to speech system." \
  -F "voice_sample=@/path/to/reference_audio.wav" \
  --output generated_speech.wav
```

#### TTS without reference audio (using default voice):

```bash
curl -X POST "http://localhost:8000/speak" \
  -F "text=Hello world, this is a test without reference audio." \
  --output generated_speech.wav
```

**Note**: When no reference audio is provided, the API uses a built-in default voice (minah1.wav) for voice characteristics.

## Development

### Local Development

For development without Docker:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Install chatterbox:
   ```bash
   git clone https://github.com/resemble-ai/chatterbox.git
   pip install -e chatterbox
   ```

3. Run locally:
   ```bash
   python main.py
   ```

### Customization

- **Change model quantization**: Modify `Dockerfile` to use different GGUF files
- **Adjust inference parameters**: Edit `main.py` temperature, cfg_weight values
- **Add new endpoints**: Extend `main.py` with additional FastAPI routes

## Model Information

- **VAE**: ve_fp32-f16.gguf (Voice Encoder, ~50MB)
- **T3**: t3_cfg-q4_k_m.gguf (Text-to-Token Transformer, 4-bit quantized, ~400MB)
- **S3Gen**: s3gen-bf16.gguf (Speech Generator, ~200MB)
- **Tokenizer**: tokenizer.json (Text tokenization, ~1MB)

Total model size: ~650MB (significantly smaller than original models)

## License

MIT License - see original Chatterbox repository for details.
