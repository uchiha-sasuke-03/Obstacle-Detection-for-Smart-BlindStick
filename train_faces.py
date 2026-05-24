"""
Face Embedding Training Script for Smart Blind Stick
=====================================================
Scans known_faces/ and builds a Facenet512 face embeddings database.

Strategy:
  - Small headshot images (< 300px): Treat the ENTIRE image as a face crop
    and extract the embedding directly (no face detection needed).
  - Larger images / group photos: Detect all faces, pick the LARGEST one.
  - Skip generic folders that don't map to a person name.
  - Store individual + averaged embeddings per person.

Usage:
    python train_faces.py
"""

import os
import sys
import pickle
import cv2
import numpy as np
import time

try:
    from deepface import DeepFace
except ImportError:
    print("ERROR: deepface is not installed. Run: pip install deepface")
    sys.exit(1)

# ── Config ──────────────────────────────────────────────────────────────────
KNOWN_FACES_DIR  = "known_faces"
DB_OUTPUT_PATH   = "known_faces/face_embeddings.pkl"
MODEL_NAME       = "Facenet512"
DETECTOR_BACKEND = "opencv"
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
SMALL_IMAGE_THRESHOLD = 300  # Images smaller than this are treated as direct face crops
MIN_FACE_PIXELS  = 30       # Minimum face size in larger images

# Folders to skip (generic names, not a person)
SKIP_FOLDERS = {"Blind Stick Face Recog", "__pycache__"}


def collect_images(root_dir):
    """Collect (image_path, person_name) pairs from known_faces/."""
    entries = []
    for item in sorted(os.listdir(root_dir)):
        item_path = os.path.join(root_dir, item)

        if os.path.isfile(item_path) and item.lower().endswith(IMAGE_EXTENSIONS):
            name = os.path.splitext(item)[0]
            entries.append((item_path, name))

        elif os.path.isdir(item_path):
            if item in SKIP_FOLDERS or item.startswith('.'):
                print(f"  [SKIP] Folder '{item}' (not a person name)")
                continue
            folder_name = item
            for fname in sorted(os.listdir(item_path)):
                fpath = os.path.join(item_path, fname)
                if os.path.isfile(fpath) and fname.lower().endswith(IMAGE_EXTENSIONS):
                    entries.append((fpath, folder_name))

    return entries


def extract_embedding_smart(image_path):
    """
    Smart face embedding extraction:
      - Small images (headshots): skip face detection, use whole image
      - Large images: detect faces, pick the LARGEST one, crop with padding
    Returns (embedding, info_string) or (None, error_string).
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, "cannot read image"

    h, w = img.shape[:2]
    max_dim = max(h, w)

    if max_dim <= SMALL_IMAGE_THRESHOLD:
        # ── SMALL HEADSHOT: use the whole image directly ──
        try:
            results = DeepFace.represent(
                img_path=img,
                model_name=MODEL_NAME,
                detector_backend="skip",  # Skip detection, treat as pre-cropped face
                enforce_detection=False,
            )
            if results:
                emb = np.array(results[0]['embedding'], dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                return emb, f"headshot ({w}x{h})"
        except Exception as e:
            return None, f"headshot error: {e}"

        return None, "headshot: no embedding"

    else:
        # ── LARGER IMAGE: detect and pick largest face ──
        try:
            faces = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
            )

            if not faces:
                # Fallback: try the whole image
                results = DeepFace.represent(
                    img_path=img,
                    model_name=MODEL_NAME,
                    detector_backend="skip",
                    enforce_detection=False,
                )
                if results:
                    emb = np.array(results[0]['embedding'], dtype=np.float32)
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                    return emb, f"fallback full ({w}x{h})"
                return None, "no faces detected"

            # Find largest face
            best_face = None
            best_area = 0
            for face_info in faces:
                region = face_info.get('facial_area', {})
                fw = region.get('w', 0)
                fh = region.get('h', 0)
                if fw < MIN_FACE_PIXELS or fh < MIN_FACE_PIXELS:
                    continue
                area = fw * fh
                if area > best_area:
                    best_area = area
                    best_face = face_info

            if best_face is None:
                # All faces too small, use the whole image
                results = DeepFace.represent(
                    img_path=img,
                    model_name=MODEL_NAME,
                    detector_backend="skip",
                    enforce_detection=False,
                )
                if results:
                    emb = np.array(results[0]['embedding'], dtype=np.float32)
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                    return emb, f"fallback full ({w}x{h})"
                return None, "faces too small"

            # Crop the largest face with 20% padding
            region = best_face['facial_area']
            x, y, fw, fh = region['x'], region['y'], region['w'], region['h']
            pad_w = int(fw * 0.2)
            pad_h = int(fh * 0.2)
            y1 = max(0, y - pad_h)
            y2 = min(h, y + fh + pad_h)
            x1 = max(0, x - pad_w)
            x2 = min(w, x + fw + pad_w)
            face_crop = img[y1:y2, x1:x2]

            if face_crop.size == 0:
                return None, "empty crop"

            results = DeepFace.represent(
                img_path=face_crop,
                model_name=MODEL_NAME,
                detector_backend="skip",  # Already cropped, skip detection
                enforce_detection=False,
            )

            if results:
                emb = np.array(results[0]['embedding'], dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                return emb, f"largest face ({fw}x{fh} in {w}x{h})"

        except Exception as e:
            return None, f"error: {e}"

        return None, "extraction failed"


def train():
    print("=" * 62)
    print("  SMART BLIND STICK - FACE TRAINING PIPELINE v2")
    print("=" * 62)
    print(f"  Model         : {MODEL_NAME} (512-dimensional)")
    print(f"  Detector      : {DETECTOR_BACKEND}")
    print(f"  Headshot mode : images <= {SMALL_IMAGE_THRESHOLD}px")
    print(f"  Min face size : {MIN_FACE_PIXELS}px (for larger images)")
    print(f"  Source dir    : {os.path.abspath(KNOWN_FACES_DIR)}")
    print("=" * 62)

    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"ERROR: '{KNOWN_FACES_DIR}' not found.")
        sys.exit(1)

    print("\nScanning images...")
    entries = collect_images(KNOWN_FACES_DIR)
    if not entries:
        print("ERROR: No images found.")
        sys.exit(1)

    people = {}
    for path, name in entries:
        people.setdefault(name, []).append(path)

    print(f"\nFound {len(entries)} images for {len(people)} people:")
    for name, paths in people.items():
        print(f"  >> {name}: {len(paths)} image(s)")

    print(f"\n--- Extracting Facenet512 embeddings ---\n")
    database = {}
    success_count = 0
    fail_count = 0
    start_time = time.time()

    for i, (image_path, name) in enumerate(entries, 1):
        short = os.path.basename(image_path)
        if len(short) > 45:
            short = short[:42] + "..."
        print(f"  [{i:2d}/{len(entries)}] {name}/{short} ... ", end="", flush=True)

        emb, info = extract_embedding_smart(image_path)
        if emb is not None:
            database.setdefault(name, []).append(emb)
            success_count += 1
            print(f"OK - {info}")
        else:
            fail_count += 1
            print(f"SKIP - {info}")

    elapsed = time.time() - start_time

    if not database:
        print("\nERROR: No faces extracted. Check your images.")
        sys.exit(1)

    # Compute per-person average
    print(f"\nComputing per-person averaged embeddings...")
    final_db = {}
    for name, emb_list in database.items():
        avg = np.mean(emb_list, axis=0).astype(np.float32)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm
        final_db[name] = {
            'embeddings': emb_list,
            'average': avg,
            'count': len(emb_list),
        }

    # Save
    output = {
        'model': MODEL_NAME,
        'detector': DETECTOR_BACKEND,
        'people': final_db,
        'trained_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(DB_OUTPUT_PATH, 'wb') as f:
        pickle.dump(output, f)

    size_kb = os.path.getsize(DB_OUTPUT_PATH) / 1024

    print(f"\n{'=' * 62}")
    print(f"  TRAINING COMPLETE")
    print(f"{'=' * 62}")
    print(f"  People registered : {len(final_db)}")
    print(f"  Total embeddings  : {success_count}")
    print(f"  Skipped images    : {fail_count}")
    print(f"  Time              : {elapsed:.1f}s")
    print(f"  DB size           : {size_kb:.1f} KB")
    print(f"  Saved to          : {DB_OUTPUT_PATH}")
    print(f"{'=' * 62}")
    for name, data in final_db.items():
        print(f"  [OK] {name}: {data['count']} embedding(s)")
    print(f"\nRestart main.py to use the new face database.")


if __name__ == '__main__':
    train()
