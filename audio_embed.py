import numpy as np
import soundfile as sf
import librosa

def embed_audio(cover_audio_path, secret_audio_path, output_path):
    cover, sr = sf.read(cover_audio_path, always_2d=False)
    secret, sr2 = sf.read(secret_audio_path, always_2d=False)

    if cover.ndim > 1:
        cover = cover.mean(axis=1)
    if secret.ndim > 1:
        secret = secret.mean(axis=1)

    if sr2 != sr:
        secret = librosa.resample(secret.astype(float), orig_sr=sr2, target_sr=sr)

    min_len = min(len(cover), len(secret))
    cover = cover[:min_len]
    secret = secret[:min_len]

    cover_i16  = np.round(cover * 32767).astype(np.int16)
    secret_i16 = np.round(secret * 32767).astype(np.int16)

    cover_u16  = cover_i16.astype(np.uint16)
    secret_u16 = secret_i16.astype(np.uint16)

    # Keep high 8 bits of cover, store high 8 bits of secret in low 8 bits
    stego_u16 = (cover_u16 & 0xFF00) | (secret_u16 >> 8)

    stego_i16 = stego_u16.astype(np.int16)
    stego = stego_i16.astype(np.float32) / 32767.0
    sf.write(output_path, stego, sr)
    print("✅ Stego audio written to", output_path)