import os
import io
import torch
import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional

from chatterbox.tts import ChatterboxTTS
from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.models.t3 import T3
from chatterbox.models.s3gen import S3Gen
from chatterbox.models.tokenizers import EnTokenizer
from safetensors.torch import load_file

app = FastAPI()

MODEL_DIR = "/app/models/box"
TOKENIZER_PATH = "/app/tokenizer.json"
VAE_ST = os.path.join(MODEL_DIR, "ve_fp32-f16-f32.safetensors")
T3_ST = os.path.join(MODEL_DIR, "t3_cfg-q4_k_m-f32.safetensors")
S3GEN_ST = os.path.join(MODEL_DIR, "s3gen-bf16-f32.safetensors")

def load_chatterbox_model(device="cpu"):
    ve = VoiceEncoder()
    ve.load_state_dict(load_file(VAE_ST))
    ve.eval()

    t3 = T3()
    t3_state = load_file(T3_ST)
    if "model" in t3_state.keys():
        t3_state = t3_state["model"][0]
    t3.load_state_dict(t3_state)
    t3.eval()

    s3gen = S3Gen()
    s3gen.load_state_dict(load_file(S3GEN_ST), strict=False)
    s3gen.eval()

    tokenizer = EnTokenizer(TOKENIZER_PATH)

    model = ChatterboxTTS(
        t3=t3,
        s3gen=s3gen,
        ve=ve,
        tokenizer=tokenizer,
        device=device
    )
    return model

def create_default_reference_audio():
    """Create a synthetic reference audio for default voice conditioning"""
    # Use the provided minah1.wav file as default reference
    default_path = "/app/minah1.wav"
    if os.path.exists(default_path):
        print(f"Using default reference audio: {default_path}")
        return default_path
    
    print("minah1.wav not found, creating synthetic reference audio...")
    # Fallback: create a simple sine wave at 440Hz (A4 note) for 1 second
    sample_rate = 22050  # S3GEN_SR
    duration = 1.0
    frequency = 440.0
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Create a simple tone with some harmonics to make it more voice-like
    audio = (np.sin(2 * np.pi * frequency * t) * 0.3 + 
             np.sin(2 * np.pi * frequency * 2 * t) * 0.2 +
             np.sin(2 * np.pi * frequency * 3 * t) * 0.1)
    
    # Add some envelope to make it more natural
    envelope = np.exp(-t * 2)  # Exponential decay
    audio = audio * envelope
    
    # Save to temporary file
    temp_path = "/tmp/default_ref.wav"
    sf.write(temp_path, audio.astype(np.float32), sample_rate)
    print(f"Created synthetic reference audio: {temp_path}")
    return temp_path

# Load the model and create default reference at startup
print("Loading Chatterbox TTS model...")
tts_model = load_chatterbox_model()
print("Creating default reference audio...")
default_ref_path = create_default_reference_audio()
print(f"Default reference path: {default_ref_path}")

# Debug: Check if minah1.wav exists
minah_path = "/app/minah1.wav"
if os.path.exists(minah_path):
    print(f"✓ minah1.wav found at {minah_path}, size: {os.path.getsize(minah_path)} bytes")
else:
    print("✗ minah1.wav NOT found at /app/minah1.wav")

@app.post("/speak", response_class=StreamingResponse)
async def speak(
    text: str = Form(...),
    voice_sample: Optional[UploadFile] = File(None)
):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text input is empty.")

    try:
        # Save uploaded reference audio to a temp file if provided
        ref_audio_path = None
        if voice_sample:
            print(f"Using uploaded voice sample: {voice_sample.filename}")
            contents = await voice_sample.read()
            ref_audio_path = "/tmp/ref.wav"
            with open(ref_audio_path, "wb") as f:
                f.write(contents)
        else:
            # Use default reference audio (minah1.wav or synthetic)
            print("No voice sample provided, using default reference audio")
            ref_audio_path = default_ref_path

        print(f"Using reference audio: {ref_audio_path}")
        
        # Verify the reference file exists and is readable
        if not os.path.exists(ref_audio_path):
            raise HTTPException(status_code=500, detail=f"Reference audio file not found: {ref_audio_path}")
        
        # Check file size and contents
        file_size = os.path.getsize(ref_audio_path)
        print(f"Reference audio file size: {file_size} bytes")
        
        # Try to read the audio file to validate it
        try:
            data, sr = sf.read(ref_audio_path)
            print(f"Audio file validation: {len(data)} samples at {sr} Hz")
        except Exception as audio_error:
            print(f"Audio file validation failed: {audio_error}")
            raise HTTPException(status_code=500, detail=f"Invalid audio file: {audio_error}")

        # Prepare conditionals and generate speech
        print("Preparing conditionals...")
        try:
            tts_model.prepare_conditionals(ref_audio_path, exaggeration=0.5)
            print("Conditionals prepared successfully")
        except Exception as cond_error:
            print(f"Error preparing conditionals: {cond_error}")
            print(f"Error type: {type(cond_error)}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to prepare conditionals: {cond_error}")
        
        print("Generating speech...")
        audio_data = tts_model.generate(
            text=text,
            temperature=0.8,
            cfg_weight=0.5
        )

        # Convert tensor to numpy and save as WAV
        if hasattr(audio_data, 'numpy'):
            audio_data = audio_data.numpy()
        audio_data = audio_data.squeeze().astype('float32')
        wav_bytes_io = io.BytesIO()
        sf.write(wav_bytes_io, audio_data, tts_model.sr, format='WAV')
        wav_bytes_io.seek(0)

        print("Speech generation completed successfully")
        return StreamingResponse(
            wav_bytes_io,
            media_type="audio/wav",
            headers={"Content-Disposition": "inline; filename=output.wav"}
        )

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        print(f"Error in TTS generation: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
