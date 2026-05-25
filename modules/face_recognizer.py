"""
Face Recognizer Module - PyTorch FaceNet + MTCNN (v5)
======================================================
Uses MTCNN for robust face detection and InceptionResnetV1 (FaceNet)
for 512-dimensional embeddings. Matches faces using cosine similarity.

Key design choices:
  - MTCNN: handles low light, angles, partial occlusion — far better than Haar
  - FaceNet (VGGFace2): 512-dim embeddings, state-of-the-art accuracy
  - Cosine similarity threshold: 0.55 (tuned for real-world conditions)
  - identify_person(): integrates directly with YOLO "person" detection
  - CLAHE fallback: enhances low-light images before re-trying detection
  - 5-second cooldown on name announcements
"""

import os
import pickle
import threading
import time
import cv2
import numpy as np
import torch
from PIL import Image

# Try to import facenet-pytorch
try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False
    print("Warning: facenet-pytorch not installed. Facial recognition disabled.")
    print("  Install with: pip install facenet-pytorch --no-deps")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FaceRecognizer:
    # ── Configuration ───────────────────────────────────────────────────
    COSINE_THRESHOLD     = 0.50        # Minimum cosine similarity for a match
    DB_FILENAME          = "face_embeddings_v5.pkl"

    def __init__(self, known_faces_dir="known_faces"):
        self.known_faces_dir = known_faces_dir
        self.is_processing = False
        self.last_seen = ""
        self.last_seen_time = 0
        self.frames_since_last_seen = 0

        # Cached name for fast lookups from YOLO loop
        self._current_name = None
        self._current_name_time = 0
        self._name_lock = threading.Lock()

        # Models
        self.mtcnn = None
        self.resnet = None
        self.database = {}
        self.db_loaded = False

        if not FACE_REC_AVAILABLE:
            return

        if not os.path.exists(known_faces_dir):
            os.makedirs(known_faces_dir)

        self._load_models()
        self._load_database()

    def _load_models(self):
        """Load MTCNN and InceptionResnetV1."""
        try:
            self.mtcnn = MTCNN(
                image_size=160,
                margin=20,
                min_face_size=20,
                thresholds=[0.5, 0.6, 0.6],
                factor=0.7,
                keep_all=False,       # Only return the best face
                device=DEVICE,
            )
            self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
            print("Face Models: MTCNN + FaceNet loaded successfully")
        except Exception as e:
            print(f"Face Models: Failed to load - {e}")
            self.mtcnn = None
            self.resnet = None

    def _load_database(self):
        """Load pre-computed face embeddings."""
        db_path = os.path.join(self.known_faces_dir, self.DB_FILENAME)
        if not os.path.exists(db_path):
            print("Face DB: No trained database found. Run 'python train_faces.py' first.")
            return

        try:
            with open(db_path, 'rb') as f:
                data = pickle.load(f)

            self.database = data.get('people', {})
            model = data.get('model', 'unknown')
            trained_at = data.get('trained_at', 'unknown')
            total = sum(d['count'] for d in self.database.values())
            names = ', '.join(self.database.keys())

            print(f"Face DB: Loaded {len(self.database)} people ({total} embeddings) [{model}]")
            print(f"Face DB: Trained at {trained_at}")
            print(f"Face DB: Known people: {names}")
            self.db_loaded = True

        except Exception as e:
            print(f"Face DB: Failed to load - {e}")

    @staticmethod
    def _cosine_sim(a, b):
        """Cosine similarity between two L2-normalized vectors."""
        return float(np.dot(a, b))

    def _get_embedding(self, img_bgr):
        """
        Detect the best face in a BGR image and return its embedding.
        Tries CLAHE enhancement as fallback for low-light.
        Returns 512-dim numpy array or None.
        """
        if self.mtcnn is None or self.resnet is None:
            return None

        # Try original image first
        emb = self._try_get_embedding(img_bgr)
        if emb is not None:
            return emb

        # Fallback: CLAHE enhancement for low light
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return self._try_get_embedding(enhanced)

    def _try_get_embedding(self, img_bgr):
        """Try to detect a face and get its embedding. Returns numpy array or None."""
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            # MTCNN returns a face tensor (already aligned + resized to 160x160)
            face_tensor = self.mtcnn(pil_img)

            if face_tensor is None:
                return None

            # If keep_all=False, face_tensor is shape (3, 160, 160)
            if face_tensor.dim() == 3:
                face_tensor = face_tensor.unsqueeze(0)

            face_tensor = face_tensor.to(DEVICE)

            with torch.no_grad():
                embedding = self.resnet(face_tensor)

            emb_np = embedding[0].cpu().numpy().astype(np.float32)

            # L2 normalize
            norm = np.linalg.norm(emb_np)
            if norm > 0:
                emb_np = emb_np / norm

            return emb_np

        except Exception:
            return None

    def _find_best_match(self, live_emb):
        """
        Compare a live embedding against the database.
        Returns (name, score) or (None, 0).

        Strategy:
          - Check average embedding first (fast rejection)
          - If promising, check individual embeddings
          - Score = 40% average + 60% best individual
        """
        best_name = None
        best_score = -1.0

        for name, data in self.database.items():
            avg_score = self._cosine_sim(live_emb, data['average'])

            if avg_score > self.COSINE_THRESHOLD * 0.75:
                individual_scores = [
                    self._cosine_sim(live_emb, emb)
                    for emb in data['embeddings']
                ]
                best_individual = max(individual_scores)
                combined = 0.40 * avg_score + 0.60 * best_individual
            else:
                combined = avg_score

            if combined > best_score:
                best_score = combined
                best_name = name

        if best_score >= self.COSINE_THRESHOLD:
            return best_name, best_score

        return None, 0

    def identify_person(self, frame, xyxy):
        """
        Called from YOLO loop when a "person" is detected.
        Crops the person region, detects a face, and matches it.

        Args:
            frame: Full BGR frame
            xyxy: YOLO bounding box [x1, y1, x2, y2]

        Returns:
            Person's name (str) if recognized, or None.
        """
        if not FACE_REC_AVAILABLE or not self.db_loaded:
            return None

        try:
            x1 = max(0, int(xyxy[0]))
            y1 = max(0, int(xyxy[1]))
            x2 = min(frame.shape[1], int(xyxy[2]))
            y2 = min(frame.shape[0], int(xyxy[3]))

            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                return None

            # Get face embedding from the person crop
            emb = self._get_embedding(person_crop)
            if emb is None:
                return None

            # Match against database
            name, score = self._find_best_match(emb)
            return name

        except Exception:
            return None

    def recognize(self, frame, tts_manager):
        """
        Standalone face recognition on the full frame.
        Non-blocking: spawns a background thread.
        """
        if not FACE_REC_AVAILABLE or self.is_processing or not self.db_loaded:
            return

        def _process():
            self.is_processing = True
            try:
                emb = self._get_embedding(frame)
                if emb is None:
                    return

                name, score = self._find_best_match(emb)
                if name is None:
                    return

                # Cache the result
                with self._name_lock:
                    self._current_name = name
                    self._current_name_time = time.time()

                # Speak the name
                now = time.time()
                if name != self.last_seen or (now - self.last_seen_time) > 5:
                    tts_manager.speak(f"I see {name}")
                    self.last_seen = name
                    self.last_seen_time = now
                    self.frames_since_last_seen = 0

            except Exception:
                pass
            finally:
                self.is_processing = False

        self.frames_since_last_seen += 1
        if self.frames_since_last_seen > 150:
            self.last_seen = ""

        threading.Thread(target=_process, daemon=True).start()

    def get_cached_name(self):
        """Returns the most recently recognized name if fresh (< 3 seconds)."""
        with self._name_lock:
            if self._current_name and (time.time() - self._current_name_time) < 3:
                return self._current_name
        return None


face_rec = FaceRecognizer()
