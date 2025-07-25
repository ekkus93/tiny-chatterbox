import os
import io
import uuid
import tempfile
import torch
import numpy as np
import soundfile as sf
import re
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional, List

from chatterbox.tts import ChatterboxTTS
from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.models.t3 import T3
from chatterbox.models.s3gen import S3Gen
from chatterbox.models.tokenizers import EnTokenizer
from chatterbox.quantization.quantize import load_quantized_model
from safetensors.torch import load_file

app = FastAPI()

MODEL_DIR = "/app/models/box"
TOKENIZER_PATH = "/app/tokenizer.json"
VAE_ST = os.path.join(MODEL_DIR, "ve.safetensors")

# Debug: Check if default voice exists
default_voice_path = "/app/voice_samples/minah1.wav"
if os.path.exists(default_voice_path):
    print(f"✓ default voice found at {default_voice_path}, size: {os.path.getsize(default_voice_path)} bytes")
else:
    print(f"✗ default voice NOT found at {default_voice_path}")

# Dynamic model loading based on what's available
def get_model_path(base_name):
    """Get the best available model (quantized if available, original otherwise)"""
    quantized_path = os.path.join(MODEL_DIR, f"{base_name}_int8.pt")
    original_path = os.path.join(MODEL_DIR, f"{base_name}_original.safetensors")
    
    if os.path.exists(quantized_path):
        return quantized_path, "quantized"
    elif os.path.exists(original_path):
        return original_path, "original"
    else:
        raise FileNotFoundError(f"No model found for {base_name}")

def load_chatterbox_model(device="cpu"):
    # Load voice encoder
    ve = VoiceEncoder()
    ve.load_state_dict(load_file(VAE_ST))
    ve.eval()

    # Load quantized T3 model using the proper quantization support
    print("Loading quantized T3 model...")
    t3, _ = load_quantized_model(T3, os.path.join(MODEL_DIR, "t3_cfg_int8.pt"), device)
    t3.to(device).eval()

    # Load original S3Gen model (NOT quantized)
    print("Loading original S3Gen model...")
    s3gen = S3Gen()
    s3gen.load_state_dict(load_file("/app/s3gen.safetensors"), strict=False)
    s3gen.to(device).eval()

    tokenizer = EnTokenizer(TOKENIZER_PATH)

    model = ChatterboxTTS(
        t3=t3,
        s3gen=s3gen,
        ve=ve,
        tokenizer=tokenizer,
        device=device
    )
    return model

# Load the model and create default reference at startup
print("Loading Chatterbox TTS model...")
tts_model = load_chatterbox_model()
print("Creating default reference audio...")
print(f"Default voice path: {default_voice_path}")

# Add this function to handle in-memory audio processing
def process_in_memory_audio(audio_bytes, format_hint=None):
    """Process audio data directly from memory without writing to disk"""
    # Default to WAV if no format hint provided
    format_hint = format_hint or 'wav'
    
    # Use BytesIO to create a file-like object in memory
    audio_buffer = io.BytesIO(audio_bytes)
    
    try:
        # Use soundfile to read directly from the in-memory buffer
        data, sample_rate = sf.read(audio_buffer, format=format_hint)
        return data, sample_rate
    except Exception as e:
        print(f"Error reading in-memory audio: {e}")
        raise e

# Extend ChatterboxTTS with a method to handle in-memory audio
def add_prepare_from_array_method():
    """Monkey patch ChatterboxTTS to handle audio arrays directly"""
    original_prepare_conditionals = ChatterboxTTS.prepare_conditionals
    
    def prepare_from_array(self, audio_array, sr, exaggeration=0.0):
        """Process audio directly from a numpy array with sample rate"""
        # Create a temporary file in memory
        with io.BytesIO() as buf:
            # Write the audio data to the in-memory buffer
            sf.write(buf, audio_array, sr, format='WAV')
            buf.seek(0)
            
            # Create a temporary file path just for the API
            temp_path = f"memory://audio_{uuid.uuid4()}.wav"
            
            # Store the audio data for this path
            if not hasattr(self, '_memory_files'):
                self._memory_files = {}
            self._memory_files[temp_path] = (buf.read(), sr)
            
            # Call the original method with our virtual path
            return original_prepare_conditionals(self, temp_path, exaggeration)
    
    # Save the original method
    if not hasattr(ChatterboxTTS, '_original_prepare_conditionals'):
        ChatterboxTTS._original_prepare_conditionals = original_prepare_conditionals
    
    # Replace the prepare_conditionals method with our custom version that handles the memory:// scheme
    def prepare_conditionals_wrapper(self, audio_path, exaggeration=0.0):
        if audio_path.startswith('memory://'):
            if hasattr(self, '_memory_files') and audio_path in self._memory_files:
                audio_data, sr = self._memory_files[audio_path]
                # Process from the stored audio data
                audio_buffer = io.BytesIO(audio_data)
                audio_array, sr = sf.read(audio_buffer)
                return self.prepare_from_array(audio_array, sr, exaggeration)
            else:
                raise ValueError(f"Memory audio not found: {audio_path}")
        else:
            # Use the original method for file paths
            return original_prepare_conditionals(self, audio_path, exaggeration)
    
    # Add our methods to the class
    ChatterboxTTS.prepare_from_array = prepare_from_array
    ChatterboxTTS.prepare_conditionals = prepare_conditionals_wrapper

# Apply the monkey patch
add_prepare_from_array_method()

@app.post("/speak", response_class=StreamingResponse)
async def speak(
    text: str = Form(...),
    voice_sample: Optional[UploadFile] = File(None),
    voice_sample_name: Optional[str] = Form(None)
):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text input is empty.")

    temp_dir = None
    
    try:
        # Only create a temporary directory if absolutely needed
        temp_dir = tempfile.TemporaryDirectory()
        
        # Determine the reference audio source
        ref_audio_path = None
        in_memory_audio = None
        
        if voice_sample:
            print(f"Using uploaded voice sample: {voice_sample.filename}")
            contents = await voice_sample.read()
            
            try:
                # Process the audio in memory
                format_hint = os.path.splitext(voice_sample.filename)[1][1:].lower() if voice_sample.filename else 'wav'
                audio_data, sr = process_in_memory_audio(contents, format_hint)
                print(f"Successfully loaded in-memory audio: {len(audio_data)} samples at {sr}Hz")
                
                # Generate a virtual path for this audio
                in_memory_path = f"memory://uploaded_{uuid.uuid4()}.wav"
                
                # Store the audio data in the model
                if not hasattr(tts_model, '_memory_files'):
                    tts_model._memory_files = {}
                tts_model._memory_files[in_memory_path] = (contents, sr)
                
                ref_audio_path = in_memory_path
                print(f"Using in-memory audio reference: {ref_audio_path}")
            except Exception as e:
                print(f"Failed to process audio in memory: {e}. Falling back to disk.")
                # Fall back to disk-based processing
                original_ext = os.path.splitext(voice_sample.filename)[1].lower() if voice_sample.filename else ".wav"
                if not original_ext or original_ext not in ['.wav', '.mp3', '.flac', '.ogg']:
                    original_ext = ".wav"
                    
                temp_file = tempfile.NamedTemporaryFile(delete=False, 
                                                       suffix=original_ext, 
                                                       dir=temp_dir.name)
                temp_file.write(contents)
                temp_file.close()
                ref_audio_path = temp_file.name
                print(f"Saved audio to temporary file: {ref_audio_path}")
        elif voice_sample_name:
            # Validate voice_sample_name to prevent path traversal
            if not re.match(r'^[a-zA-Z0-9_\-.]+\.(wav|mp3|flac|ogg)$', voice_sample_name):
                raise HTTPException(
                    status_code=400, 
                    detail="Invalid voice sample name. Must be alphanumeric with extensions .wav, .mp3, .flac, or .ogg"
                )
            
            # Construct safe path to the voice sample
            safe_path = os.path.join("/app/voice_samples", voice_sample_name)
            
            # Ensure the path doesn't escape the voice_samples directory
            if not os.path.abspath(safe_path).startswith("/app/voice_samples/"):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid voice sample path"
                )
            
            if not os.path.exists(safe_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"Voice sample '{voice_sample_name}' not found"
                )
                
            print(f"Using voice sample from library: {voice_sample_name}")
            ref_audio_path = safe_path
        else:
            # Use default reference audio (minah1.wav or synthetic)
            print("No voice sample provided, using default reference audio")
            ref_audio_path = default_voice_path
        print(f"Reference audio path: {ref_audio_path}")

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
        raise
    except Exception as e:
        print(f"Error in TTS generation: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")
    finally:
        # Clean up temporary directory and all files inside it
        if temp_dir is not None:
            try:
                temp_dir.cleanup()
                print(f"Cleaned up temporary directory")
            except Exception as cleanup_error:
                print(f"Warning: Failed to clean up temporary directory: {cleanup_error}")

@app.get("/list_voices", response_class=JSONResponse)
async def list_voices():
    """
    Return a list of all available voice sample files in the /app/voice_samples directory.
    Each entry includes filename, size in bytes, and file extension.
    """
    voice_samples_dir = "/app/voice_samples"
    
    try:
        if not os.path.exists(voice_samples_dir):
            return JSONResponse(
                status_code=404,
                content={"error": f"Voice samples directory not found: {voice_samples_dir}"}
            )
        
        # Get all files in the directory
        files = os.listdir(voice_samples_dir)
        
        # Filter for audio files and collect metadata
        voice_files = []
        for filename in files:
            filepath = os.path.join(voice_samples_dir, filename)
            if os.path.isfile(filepath):
                file_ext = os.path.splitext(filename)[1].lower()
                # Include common audio formats
                if file_ext in ['.wav', '.mp3', '.flac', '.ogg']:
                    voice_files.append({
                        "filename": filename,
                        "size_bytes": os.path.getsize(filepath),
                        "extension": file_ext
                    })
        
        return {"voices": voice_files}
        
    except Exception as e:
        print(f"Error listing voice files: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to list voice files: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

