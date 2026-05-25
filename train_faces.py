"""
Face Embedding Training Script (v5 - PyTorch FaceNet + MTCNN)
==============================================================
Uses MTCNN for face detection and InceptionResnetV1 (FaceNet) for
512-dimensional face embeddings. Runs on PyTorch — no TensorFlow needed.

Strategy:
  - MTCNN: robust face detection that works in low light, angles, etc.
  - InceptionResnetV1: pretrained on VGGFace2, produces 512-dim embeddings
  - Data augmentation: brightness/gamma/flip variants for lighting robustness
  - Stores per-person embeddings + averaged centroid for fast matching

Usage:
    python train_faces.py
"""

import os
import sys
import pickle
import cv2
import numpy as np
import time
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# ── Config ──────────────────────────────────────────────────────────────────
KNOWN_FACES_DIR  = "known_faces"
DB_OUTPUT_PATH   = "known_faces/face_embeddings_v5.pkl"
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# Folders to skip (generic names, not a person)
SKIP_FOLDERS = {"Blind Stick Face Recog", "__pycache__"}

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_models():
    """Initialize MTCNN and InceptionResnetV1."""
    print(f"  Loading MTCNN face detector...")
    mtcnn = MTCNN(
        image_size=160,
        margin=20,
        min_face_size=20,
        thresholds=[0.5, 0.6, 0.6],  # More lenient for low-light
        factor=0.7,
        keep_all=True,
        device=DEVICE,
    )

    print(f"  Loading InceptionResnetV1 (VGGFace2)...")
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

    return mtcnn, resnet


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
                print(f"  [SKIP] Folder '{item}'")
                continue
            folder_name = item
            for fname in sorted(os.listdir(item_path)):
                fpath = os.path.join(item_path, fname)
                if os.path.isfile(fpath) and fname.lower().endswith(IMAGE_EXTENSIONS):
                    entries.append((fpath, folder_name))

    return entries


def augment_image(img_bgr):
    """
    Generate augmented versions of an image for lighting robustness.
    Returns list of BGR images (including original).
    """
    augmented = [img_bgr]

    # Horizontal flip
    augmented.append(cv2.flip(img_bgr, 1))

    # Gamma corrections (simulate different lighting)
    for gamma in [0.4, 0.6, 1.5, 2.0]:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in range(256)]).astype("uint8")
        augmented.append(cv2.LUT(img_bgr, table))

    # Brightness adjustments
    for alpha, beta in [(0.6, -20), (1.4, 20)]:
        augmented.append(cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta))

    # CLAHE on the luminance channel
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    augmented.append(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))

    return augmented


def extract_embedding(img_bgr, mtcnn, resnet):
    """
    Detect the largest face in a BGR image and return its 512-dim embedding.
    Returns (embedding_np, info_str) or (None, error_str).
    """
    # Convert BGR to RGB PIL Image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    # Detect faces with MTCNN — returns face tensors ready for the model
    faces, probs = mtcnn(pil_img, return_prob=True)

    if faces is None or len(faces) == 0:
        # Fallback: try with CLAHE enhanced image
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        img_rgb2 = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        pil_img2 = Image.fromarray(img_rgb2)
        faces, probs = mtcnn(pil_img2, return_prob=True)

        if faces is None or len(faces) == 0:
            return None, "no face detected"

    # Pick the face with highest probability
    best_idx = int(np.argmax(probs))
    face_tensor = faces[best_idx].unsqueeze(0).to(DEVICE)

    # Get embedding
    with torch.no_grad():
        embedding = resnet(face_tensor)

    emb_np = embedding[0].cpu().numpy().astype(np.float32)

    # L2 normalize
    norm = np.linalg.norm(emb_np)
    if norm > 0:
        emb_np = emb_np / norm

    return emb_np, f"prob={probs[best_idx]:.3f}"


def train():
    print("=" * 62)
    print("  SMART BLIND STICK - FACE TRAINING v5 (PyTorch FaceNet)")
    print("=" * 62)
    print(f"  Model         : InceptionResnetV1 (VGGFace2, 512-dim)")
    print(f"  Detector      : MTCNN (PyTorch)")
    print(f"  Device        : {DEVICE}")
    print(f"  Augmentation  : flip + gamma + brightness + CLAHE (9x)")
    print(f"  Source dir    : {os.path.abspath(KNOWN_FACES_DIR)}")
    print("=" * 62)

    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"ERROR: '{KNOWN_FACES_DIR}' not found.")
        sys.exit(1)

    # Initialize models
    mtcnn, resnet = init_models()

    print("\nScanning images...")
    entries = collect_images(KNOWN_FACES_DIR)
    if not entries:
        print("ERROR: No images found.")
        sys.exit(1)

    # Group by person
    people = {}
    for path, name in entries:
        people.setdefault(name, []).append(path)

    print(f"\nFound {len(entries)} images for {len(people)} people:")
    for name, paths in people.items():
        print(f"  >> {name}: {len(paths)} image(s)")

    print(f"\n--- Extracting FaceNet embeddings ---\n")
    database = {}
    success_count = 0
    fail_count = 0
    start_time = time.time()

    for i, (image_path, name) in enumerate(entries, 1):
        short = os.path.basename(image_path)
        if len(short) > 40:
            short = short[:37] + "..."
        print(f"  [{i:2d}/{len(entries)}] {name}/{short}")

        img = cv2.imread(image_path)
        if img is None:
            fail_count += 1
            print(f"           SKIP - cannot read")
            continue

        # Generate augmented versions
        augmented_images = augment_image(img)
        img_success = 0

        for j, aug_img in enumerate(augmented_images):
            emb, info = extract_embedding(aug_img, mtcnn, resnet)
            if emb is not None:
                database.setdefault(name, []).append(emb)
                img_success += 1

        if img_success > 0:
            success_count += 1
            print(f"           OK - {img_success}/{len(augmented_images)} variants embedded")
        else:
            fail_count += 1
            print(f"           SKIP - no face in any variant")

    elapsed = time.time() - start_time

    if not database:
        print("\nERROR: No faces extracted. Check your images.")
        sys.exit(1)

    # Compute per-person average centroid
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
        'model': 'InceptionResnetV1-VGGFace2',
        'embedding_dim': 512,
        'people': final_db,
        'trained_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(DB_OUTPUT_PATH, 'wb') as f:
        pickle.dump(output, f)

    size_kb = os.path.getsize(DB_OUTPUT_PATH) / 1024
    total_embs = sum(d['count'] for d in final_db.values())

    print(f"\n{'=' * 62}")
    print(f"  TRAINING COMPLETE")
    print(f"{'=' * 62}")
    print(f"  People registered : {len(final_db)}")
    print(f"  Total embeddings  : {total_embs}")
    print(f"  Source images OK  : {success_count}")
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
