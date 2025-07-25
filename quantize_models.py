import torch
import torch.quantization as quant
from chatterbox.models.t3 import T3
from chatterbox.models.voice_encoder import VoiceEncoder
from safetensors.torch import load_file, save_file

def quantize_models():
    print('Starting model quantization...')

    # Voice Encoder - copy as-is (small model, quantization not beneficial)
    print('Processing Voice Encoder...')
    ve = VoiceEncoder()
    ve.load_state_dict(load_file('/app/ve.safetensors'))
    ve.eval()
    save_file(ve.state_dict(), '/app/models/box/ve.safetensors')
    print('Voice Encoder copied')

    # T3 - Apply PyTorch's native dynamic quantization (simpler and more reliable)
    print('Quantizing T3 model...')
    t3 = T3()
    t3_state = load_file('/app/t3_cfg.safetensors')
    t3_state = t3_state['model'][0] if 'model' in t3_state else t3_state
    t3.load_state_dict(t3_state)
    t3.eval()

    # Dynamic quantization - only quantize Linear layers (preserves embeddings)
    print('Applying dynamic quantization to T3...')
    t3_quantized = quant.quantize_dynamic(t3, {torch.nn.Linear}, dtype=torch.qint8)
    
    # Restore the device property that dynamic quantization breaks
    def device_property(self):
        try:
            return self.speech_head.weight().device
        except:
            # Fallback: find any parameter and return its device
            for param in self.parameters():
                if torch.is_tensor(param):
                    return param.device
            return torch.device('cpu')
    
    # Override the device property on the class
    t3_quantized.__class__.device = property(device_property)
    
    # Test that the device property works correctly
    print('Testing quantized T3 device property...')
    try:
        device_test = t3_quantized.device
        print(f'✓ T3 device property works: {device_test}')
    except Exception as e:
        print(f'✗ T3 device property test failed: {e}')
        raise RuntimeError(f'Quantized T3 model device property is broken: {e}')
    
    # Save as complete model (not state dict due to quantization changes)
    torch.save(t3_quantized, '/app/models/box/t3_cfg_int8.pt')
    print('T3 quantized and saved')

    print('Model quantization completed!')

if __name__ == "__main__":
    quantize_models()
