"""
Face Recognizer Module - Pre-trained Facenet512 Matching
=========================================================
Loads a pre-built face embeddings database (from train_faces.py) and matches
live webcam faces against it using cosine similarity for fast, accurate,
real-time recognition. Speaks the person's name via TTS.

Key design choices:
  - Facenet512: 512-dim vectors, most accurate model in DeepFace
  - Only the LARGEST face in the frame is matched (the person closest to camera)
  - Weighted scoring: 60% best individual match + 40% average match
  - Cosine similarity threshold: 0.55 (tuned for real-world conditions)
  - 3-second cooldown on repeated name announcements
"""

import os
import pickle
import threading
import time
import numpy as np

try:
    from deepface import DeepFace
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False
    print("Warning: deepface library not installed. Facial recognition disabled.")


class FaceRecognizer:
    # ── Configuration ───────────────────────────────────────────────────
    MODEL_NAME          = "Facenet512"
    DETECTOR_BACKEND    = "opencv"
    COSINE_THRESHOLD    = 0.55       # Minimum cosine similarity for a match
    MIN_FACE_PIXELS     = 30         # Ignore tiny faces in live feed
    DB_FILENAME         = "face_embeddings.pkl"

    def __init__(self, known_faces_dir="known_faces"):
        self.known_faces_dir = known_faces_dir
        self.is_processing = False
        self.last_seen = ""
        self.last_seen_time = 0
        self.frames_since_last_seen = 0

        # Pre-trained database
        self.database = {}    # {name: {'embeddings': [...], 'average': np.array}}
        self.db_loaded = False

        if not FACE_REC_AVAILABLE:
            return

        if not os.path.exists(known_faces_dir):
            os.makedirs(known_faces_dir)
            print(f"Created '{known_faces_dir}' directory.")

        self._load_database()

    def _load_database(self):
        """Load pre-computed face embeddings from the pickle database."""
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

    def _get_largest_face_embedding(self, frame):
        """
        Detect faces in the live frame, pick the LARGEST one (closest person),
        and return its Facenet512 embedding (L2 normalized).
        """
        try:
            # Detect all faces
            faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=self.DETECTOR_BACKEND,
                enforce_detection=False,
            )

            if not faces:
                return None

            # Find the LARGEST face
            best_face = None
            best_area = 0
            for face_info in faces:
                region = face_info.get('facial_area', {})
                w = region.get('w', 0)
                h = region.get('h', 0)
                if w < self.MIN_FACE_PIXELS or h < self.MIN_FACE_PIXELS:
                    continue
                area = w * h
                if area > best_area:
                    best_area = area
                    best_face = face_info

            if best_face is None:
                return None

            # Crop the face with padding
            region = best_face['facial_area']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            pad_w = int(w * 0.2)
            pad_h = int(h * 0.2)
            y1 = max(0, y - pad_h)
            y2 = min(frame.shape[0], y + h + pad_h)
            x1 = max(0, x - pad_w)
            x2 = min(frame.shape[1], x + w + pad_w)
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                return None

            # Get Facenet512 embedding (skip detection since already cropped)
            results = DeepFace.represent(
                img_path=face_crop,
                model_name=self.MODEL_NAME,
                detector_backend="skip",
                enforce_detection=False,
            )

            if not results:
                return None

            emb = np.array(results[0]['embedding'], dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return emb

        except Exception:
            return None

    def _find_best_match(self, live_emb):
        """
        Compare a live face embedding against the trained database.
        Returns (name, score) or (None, 0).

        Scoring strategy:
          - For each person, compute similarity against their average embedding
          - If promising (> 80% of threshold), also check individual embeddings
          - Final score = 40% average + 60% best individual (rewards strong match)
        """
        best_name = None
        best_score = -1.0

        for name, data in self.database.items():
            # Quick check against average embedding
            avg_score = self._cosine_sim(live_emb, data['average'])

            # If average is promising, refine with individual embeddings
            if avg_score > self.COSINE_THRESHOLD * 0.80:
                individual_scores = [
                    self._cosine_sim(live_emb, emb)
                    for emb in data['embeddings']
                ]
                best_individual = max(individual_scores)
                # Weighted: favors strong individual match
                combined = 0.40 * avg_score + 0.60 * best_individual
            else:
                combined = avg_score

            if combined > best_score:
                best_score = combined
                best_name = name

        if best_score >= self.COSINE_THRESHOLD:
            return best_name, best_score

        return None, 0

    def recognize(self, frame, tts_manager):
        """
        Run face recognition on a frame (called every N frames from main loop).
        Non-blocking: spawns a background thread.
        """
        if not FACE_REC_AVAILABLE or self.is_processing or not self.db_loaded:
            return

        def _process():
            self.is_processing = True
            try:
                # Step 1: Get the largest face's embedding from the live frame
                live_emb = self._get_largest_face_embedding(frame)
                if live_emb is None:
                    return

                # Step 2: Match against database
                name, score = self._find_best_match(live_emb)

                if name is None:
                    return

                # Step 3: Speak the name (with cooldown to prevent spam)
                now = time.time()
                if name != self.last_seen or (now - self.last_seen_time) > 8:
                    tts_manager.speak(f"I see {name}")
                    self.last_seen = name
                    self.last_seen_time = now
                    self.frames_since_last_seen = 0

            except Exception:
                pass
            finally:
                self.is_processing = False

        # Slowly forget the last seen person
        self.frames_since_last_seen += 1
        if self.frames_since_last_seen > 150:
            self.last_seen = ""

        threading.Thread(target=_process, daemon=True).start()


face_rec = FaceRecognizer()
