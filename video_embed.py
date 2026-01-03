import cv2
import numpy as np
import os
import subprocess
from pathlib import Path

def _ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)

def embed_secret_video(cover_path, secret_path, output_path, tmp_dir="tmp_frames"):
    """
    Embed a secret video into a cover video using 6-bit steganography.
    Each pixel in the secret is scaled from 8-bit to 6-bit and stored in the
    lower 6 bits of the corresponding pixel in the cover.
    """
    _ensure_dir(tmp_dir)

    cap_c = cv2.VideoCapture(cover_path)
    cap_s = cv2.VideoCapture(secret_path)

    width = int(cap_c.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_c.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_c.get(cv2.CAP_PROP_FPS) or 30

    i = 0
    while True:
        ok_c, frame_c = cap_c.read()
        ok_s, frame_s = cap_s.read()

        if not ok_c:
            break

        if not ok_s:
            frame_s = np.zeros_like(frame_c)
        else:
            frame_s = cv2.resize(frame_s, (width, height))

        cover = frame_c.astype(np.uint8)
        secret = frame_s.astype(np.uint8)

        # Convert secret to 6-bit (scale 0–255 to 0–63)
        secret_6bit = (secret.astype(np.uint16) * 63) // 255
        secret_6bit = secret_6bit.astype(np.uint8)

        # Embed: keep top 2 bits of cover, insert 6-bit secret
        cover_masked = cover & 0b11000000
        stego = cover_masked | secret_6bit

        frame_file = os.path.join(tmp_dir, f"frame_{i:06d}.png")
        cv2.imwrite(frame_file, stego)  # Save as PNG to avoid compression artifacts
        i += 1

    cap_c.release()
    cap_s.release()

    # Combine frames into a video using FFmpeg (lossless)
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-framerate", str(int(fps)),
        "-i", os.path.join(tmp_dir, "frame_%06d.png"),
        "-c:v", "ffv1", output_path
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ Stego video created using FFmpeg (lossless).")
    except Exception as e:
        print("⚠ FFmpeg failed, using OpenCV fallback:", e)
        fourcc = cv2.VideoWriter_fourcc(*'DIB ')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for j in range(i):
            img = cv2.imread(os.path.join(tmp_dir, f"frame_{j:06d}.png"), cv2.IMREAD_UNCHANGED)
            out.write(img)
        out.release()
        print("✅ Stego video created using OpenCV fallback.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python embed_secret_video.py <cover_video> <secret_video> <output_stego_video>")
    else:
        embed_secret_video(sys.argv[1], sys.argv[2], sys.argv[3])

