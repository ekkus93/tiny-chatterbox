# Tiny Chatterbox TTS Docker API

A lightweight Docker container for text-to-speech synthesis using quantized GGUF models via Chatterbox TTS. Based on models from https://huggingface.co/calcuis/chatterbox-gguf.

## Features

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

**Note**: The build process will:
- Download GGUF models from HuggingFace (ve_fp32-f16.gguf, t3_cfg-q4_k_m.gguf, s3gen-bf16.gguf)
- Convert GGUF files to safetensors format during build
- Install all dependencies in a conda environment

### 2. Run the Docker Container

```bash
docker run -p 8000:8000 tiny-chatterbox
```

The API will be available at `http://localhost:8000`

**Optional**: Run with custom port mapping:
```bash
docker run -p 9000:8000 tiny-chatterbox
```

### 3. API Documentation

Once running, visit these URLs:
- **Interactive API docs**: http://localhost:8000/docs
- **ReDoc documentation**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## API Usage Examples

### Using curl

#### Basic TTS with reference audio:

```bash
curl -X POST "http://localhost:8000/speak" \
  -F "text=Hello world, this is a test of the text to speech system." \
  -F "voice_sample=@/path/to/reference_audio.wav" \
  --output generated_speech.wav
```

#### With form data from files:

```bash
curl -X POST "http://localhost:8000/speak" \
  -F "text=@text_input.txt" \
  -F "voice_sample=@reference_voice.wav" \
  --output output_speech.wav
```

### Using wget

#### Basic usage:

```bash
wget --method=POST \
  --header="Content-Type: multipart/form-data" \
  --body-data="text=Welcome to Chatterbox TTS API" \
  --post-file=reference.wav \
  "http://localhost:8000/speak" \
  -O generated_audio.wav
```

#### Advanced wget with form encoding:

```bash
# Create a temporary form file
cat > /tmp/form_data << 'EOF'
--boundary123
Content-Disposition: form-data; name="text"

This is my text to synthesize into speech.
--boundary123
Content-Disposition: form-data; name="voice_sample"; filename="reference.wav"
Content-Type: audio/wav

EOF

# Append binary audio data and closing boundary
cat reference_audio.wav >> /tmp/form_data
echo -e "\n--boundary123--" >> /tmp/form_data

# Send request
wget --post-file=/tmp/form_data \
  --header="Content-Type: multipart/form-data; boundary=boundary123" \
  "http://localhost:8000/speak" \
  -O synthesized_output.wav
```

### Using Python requests

```python
import requests

# Prepare the request
url = "http://localhost:8000/speak"
files = {
    'voice_sample': ('reference.wav', open('reference_audio.wav', 'rb'), 'audio/wav')
}
data = {
    'text': 'Hello, this is a test of the Chatterbox TTS API.'
}

# Send request
response = requests.post(url, files=files, data=data)

# Save the generated audio
if response.status_code == 200:
    with open('generated_speech.wav', 'wb') as f:
        f.write(response.content)
    print("Audio generated successfully!")
else:
    print(f"Error: {response.status_code}, {response.text}")
```

### Using JavaScript/Node.js

```javascript
const FormData = require('form-data');
const fs = require('fs');
const fetch = require('node-fetch');

async function generateSpeech() {
    const form = new FormData();
    form.append('text', 'Hello from JavaScript!');
    form.append('voice_sample', fs.createReadStream('reference_audio.wav'));

    const response = await fetch('http://localhost:8000/speak', {
        method: 'POST',
        body: form
    });

    if (response.ok) {
        const buffer = await response.buffer();
        fs.writeFileSync('generated_speech.wav', buffer);
        console.log('Audio generated successfully!');
    } else {
        console.error('Error:', response.status, await response.text());
    }
}

generateSpeech();
```

## API Endpoints

### POST /speak

Generate speech from text using a reference audio sample.

**Parameters:**
- `text` (form field, required): Text to synthesize
- `voice_sample` (file upload, required): Reference audio file (WAV format recommended)

**Response:**
- Content-Type: `audio/wav`
- Body: Generated audio file

**Example Response Headers:**
