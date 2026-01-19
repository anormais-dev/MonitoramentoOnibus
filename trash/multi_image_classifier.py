# multi_images_classifier_fixed.py
# Robust batch classifier - processa todos os arquivos em SOURCE_DIR
# Move classificados para classified/correct, classified/false
# Move com erro para classified/unknown

import os
import shutil
from pathlib import Path
import traceback

# IMPORT: adapte para o nome do seu script de classificação
# Se você usa single_image_classify_fixed.py, importe de lá:
try:
    from single_image_classifier import classify_single
except Exception:
    # fallback: se você tiver single_image_classifier.py com função classify_single
    from single_image_classifier import classify_single

SOURCE_DIR = "detections/cropped"   # pasta com os crops a classificar
OUT_CORRECT = "classified/correct"
OUT_FALSE = "classified/false"
OUT_UNKNOWN = "classified/unknown"

os.makedirs(OUT_CORRECT, exist_ok=True)
os.makedirs(OUT_FALSE, exist_ok=True)
os.makedirs(OUT_UNKNOWN, exist_ok=True)

VALID_EXT = [".jpg", ".jpeg", ".png", ".bmp"]

def is_image_file(path: Path):
    return path.is_file() and path.suffix.lower() in VALID_EXT

def batch_classify(source_dir=SOURCE_DIR):
    src = Path(source_dir)
    if not src.exists():
        print(f"[ERROR] Source dir not found: {src.resolve()}")
        return

    files = sorted([f for f in src.iterdir() if is_image_file(f)])
    if not files:
        print(f"[INFO] No image files found in {src}")
        return

    print(f"[INFO] Found {len(files)} images. Starting classification...\n")

    for f in files:
        print(f">>> Processing: {f.name}")
        try:
            result = classify_single(str(f))
        except Exception as e:
            # catch unexpected exceptions raised by classify_single
            print(f"[ERROR] classify_single raised exception for {f.name}: {e}")
            traceback.print_exc()
            dest = Path(OUT_UNKNOWN) / f.name
            shutil.copy2(str(f), str(dest))
            print(f"[INFO] Copied to unknown: {dest}\n")
            continue

        # classify_single may return None on internal error -> handle gracefully
        if not result:
            print(f"[WARN] classify_single returned None or empty for {f.name}. Moving to unknown.")
            dest = Path(OUT_UNKNOWN) / f.name
            shutil.copy2(str(f), str(dest))
            print(f"[INFO] Copied to unknown: {dest}\n")
            continue

        # expected result structure: {"predicted": "correct"|"false", ...}
        predicted = result.get("predicted")
        if not predicted:
            print(f"[WARN] No 'predicted' in result for {f.name}. Moving to unknown.")
            dest = Path(OUT_UNKNOWN) / f.name
            shutil.copy2(str(f), str(dest))
            print(f"[INFO] Copied to unknown: {dest}\n")
            continue

        predicted = str(predicted).lower().strip()
        if predicted == "correct":
            dest = Path(OUT_CORRECT) / f.name
        elif predicted == "false":
            dest = Path(OUT_FALSE) / f.name
        else:
            # unknown label
            dest = Path(OUT_UNKNOWN) / f.name

        # copy (non-destructive). Use shutil.move(...) if you prefer mover.
        try:
            shutil.copy2(str(f), str(dest))
            print(f"[INFO] {f.name} -> {dest}\n")
        except Exception as e:
            print(f"[ERROR] failed to copy {f.name} -> {dest}: {e}")
            traceback.print_exc()
            # attempt move as fallback
            try:
                shutil.move(str(f), str(dest))
                print(f"[INFO] moved as fallback: {dest}\n")
            except Exception as e2:
                print(f"[ERROR] move fallback failed: {e2}\n")

    print("=== BATCH PROCESS COMPLETE ===")

if __name__ == "__main__":
    batch_classify()
