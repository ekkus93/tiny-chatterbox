import os
import io
import torch
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

tts_model = load_chatterbox_model()

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
            contents = await voice_sample.read()
            ref_audio_path = "/tmp/ref.wav"
            with open(ref_audio_path, "wb") as f:
                f.write(contents)

        # Prepare conditionals and generate speech
        if ref_audio_path:
            tts_model.prepare_conditionals(ref_audio_path, exaggeration=0.5)
        else:
            # Use a default reference or raise error
            raise HTTPException(status_code=400, detail="Reference audio is required.")

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

        return StreamingResponse(
            wav_bytes_io,
            media_type="audio/wav",
            headers={"Content-Disposition": "inline; filename=output.wav"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
