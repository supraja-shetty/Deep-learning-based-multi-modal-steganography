import cv2
import numpy as np
import os
import tempfile
import subprocess
from pathlib import Path

def _ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)

def _frames_from_video_to_dir(video_path, out_dir):
    _ensure_dir(out_dir)
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            os.path.join(out_dir, "frame_%06d.png")
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        print("⚠ FFmpeg failed, using OpenCV fallback:", e)
        cap = cv2.VideoCapture(video_path)
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imwrite(os.path.join(out_dir, f"frame_{i:06d}.png"), frame)
            i += 1
        cap.release()
        return True

def extract_secret_video(stego_path, output_path, tmp_dir=None):
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="stego_extract_")

    frames_dir = os.path.join(tmp_dir, "frames")
    _ensure_dir(frames_dir)

    print(f"📥 Extracting frames from stego video: {stego_path}")
    _frames_from_video_to_dir(stego_path, frames_dir)

    files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
    if len(files) == 0:
        raise RuntimeError("❌ No frames found to extract.")

    out_frames_dir = os.path.join(tmp_dir, "secret_frames")
    _ensure_dir(out_frames_dir)

    print(f"🔍 Processing {len(files)} frames...")

    for idx, fname in enumerate(files):
        img_path = os.path.join(frames_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"⚠ Skipping unreadable frame: {fname}")
            continue

        extracted_6bit = img & 0b00111111
        secret_recon = (extracted_6bit.astype(np.uint16) * 255) // 63
        secret_recon = np.clip(secret_recon, 0, 255).astype(np.uint8)

        secret_recon = cv2.medianBlur(secret_recon, 3)

        out_path = os.path.join(out_frames_dir, f"secret_{idx:06d}.png")
        cv2.imwrite(out_path, secret_recon)

        if idx % 50 == 0 or idx == len(files) - 1:
            print(f"✅ Processed frame {idx + 1}/{len(files)}")

    fps = 30
    print(f"📦 Reassembling secret video to: {output_path}")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", os.path.join(out_frames_dir, "secret_%06d.png"),
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ Secret video reassembled using FFmpeg.")
    except Exception as e:
        print("⚠ FFmpeg failed, using OpenCV fallback:", e)
        sample = cv2.imread(os.path.join(out_frames_dir, "secret_000000.png"))
        h, w = sample.shape[:2]
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
        for idx in range(len(files)):
            img = cv2.imread(os.path.join(out_frames_dir, f"secret_{idx:06d}.png"))
            out.write(img)
        out.release()
        print("✅ Secret video assembled using fallback method.")

