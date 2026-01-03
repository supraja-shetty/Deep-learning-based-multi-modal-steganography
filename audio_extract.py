import numpy as np
import soundfile as sf

def extract_audio(stego_audio_path, output_path):
    stego, sr = sf.read(stego_audio_path, always_2d=False)
    if stego.ndim > 1:
        stego = stego.mean(axis=1)

    stego_i16 = np.round(stego * 32767).astype(np.int16)
    stego_u16 = stego_i16.astype(np.uint16)

    # Rebuild secret from low 8 bits (shifted back)
    secret_u16 = (stego_u16 & 0x00FF) << 8

    secret_i16 = secret_u16.astype(np.int16)
    secret = secret_i16.astype(np.float32) / 32767.0

    sf.write(output_path, secret, sr)
    print("✅ Secret audio extracted to", output_path)