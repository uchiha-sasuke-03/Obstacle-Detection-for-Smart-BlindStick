# Smart Blind Stick — AI Object Detection & Navigation System

An advanced, real-time AI computer vision system designed to act as the **"eyes"** for a smart blind stick. Powered by **YOLOv8** and **Facenet512**, this system runs locally on a camera feed to detect obstacles, analyze spatial positions, recognize known faces, read text via OCR, and analyze traffic lights — all while providing seamless **audio feedback** without blocking the video stream.

---

## 🚀 Features

### 🔍 Object Detection & Spatial Awareness
Uses a pre-trained **YOLOv8s** model to detect common objects (people, vehicles, furniture, etc.). The camera feed is divided into a **3×3 grid** to estimate:
- **Distance**: "close" or "far" (based on bounding box size)
- **Location**: "left", "straight ahead", or "right"

### 🚦 Continuous Traffic Light Detection
When a traffic light is detected, the AI zooms in, analyzes the dominant colors using **OpenCV HSV masking**, and announces:
> *"Traffic light is Red / Yellow / Green"*

### 🛑 Automatic Road Sign Reading
Detects road signs (e.g., Stop Signs) and automatically triggers the **OCR** engine to read the text written on the sign aloud.

### 📖 On-Demand Text Reading (OCR)
Point the camera at any book, document, or sign and press **`r`** to scan and read all visible text using **EasyOCR**.

### 👤 Facial Recognition (Pre-trained Facenet512)
Uses a **pre-trained Facenet512 embedding database** for fast, accurate face recognition:
- **Training Pipeline** (`train_faces.py`): Extracts 512-dimensional face embeddings from all images in `known_faces/`, supporting both individual headshots and subfolder-based grouping.
- **Runtime Matching**: Detects the **largest face** in the live frame, extracts its embedding, and compares it against the database using **weighted cosine similarity** (60% best individual match + 40% average).
- **Smart Name Extraction**: Subfolder names become person names (e.g., `known_faces/Nandeeshwar Sir/*.jpg` → "I see Nandeeshwar Sir").
- **Anti-Spam Cooldown**: 8-second cooldown prevents repeated announcements for the same person.

### 🔊 Asynchronous Audio Engine (TTS)
Uses native **Windows SAPI** (`win32com`) inside an isolated, non-blocking background thread. The video feed will *never* lag, stutter, or freeze while the computer is speaking.

### 🛡️ Smart Anti-Spam Muzzle
Prevents auditory overload by enforcing a **3-second cooldown** on repeated obstacle announcements in the same grid location, while giving high-priority exceptions to rapidly changing hazards like traffic lights.

---

## 💻 Prerequisites & Requirements

This software is built and optimized for **Windows**.

### 1. Hardware Required
- A PC/Laptop (a dedicated GPU is recommended for higher framerates).
- A standard Webcam (or Raspberry Pi Camera module).

### 2. Software Required
- **Python 3.11** (Highly recommended. Newer or older versions may face dependency compilation issues with TensorFlow/Keras).
- **Git** (to clone the repository).

---

## ⚙️ Installation Guide

### Step 1: Clone the Repository
```cmd
git clone https://github.com/uchiha-sasuke-03/Obstacle-Detection-for-Smart-BlindStick.git
cd Obstacle-Detection-for-Smart-BlindStick
```

### Step 2: Create a Virtual Environment (Recommended)
```cmd
py -3.11 -m venv venv
.\venv\Scripts\activate
```

### Step 3: Install Dependencies
```cmd
pip install -r requirements.txt
```

> **Note:** The first run will automatically download pre-trained weights for `ultralytics`, `deepface`, and `easyocr` into your system cache. This may take a few minutes.

### Step 4: Setup Facial Recognition

**Option A: Individual Photos (Simplest)**
1. Place clear headshot images directly inside the `known_faces/` folder.
2. Name each file with the person's name (e.g., `John.jpg`, `Mom.png`).

**Option B: Multiple Photos per Person (Best Accuracy)**
1. Create a subfolder inside `known_faces/` named after the person.
2. Place multiple photos of that person inside the subfolder.

```
known_faces/
├── Anupam.jpg              ← Individual headshot
├── Harsha.jpg
├── Nandeeshwar Sir/        ← Subfolder = person name
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
```

### Step 5: Train the Face Recognition Model
```cmd
python train_faces.py
```

This will:
- Scan all images in `known_faces/` (including subfolders)
- Extract **Facenet512** face embeddings (512-dimensional vectors)
- For small headshots (≤300px): use the entire image as a face crop
- For larger images: detect and extract only the **largest face** (ignores background people)
- Save the database to `known_faces/face_embeddings.pkl`

**Example output:**
```
==============================================================
  SMART BLIND STICK - FACE TRAINING PIPELINE v2
==============================================================
  Model         : Facenet512 (512-dimensional)
  Detector      : opencv
  Headshot mode : images <= 300px
==============================================================

  [ 1/14] Adithya/Adithya.jpg ... OK - headshot (153x153)
  [ 2/14] Anupam/Anupam.jpg ... OK - headshot (137x137)
  [ 5/14] Nandeeshwar Sir/photo1.jpeg ... OK - largest face (227x227 in 720x1600)
  ...

==============================================================
  TRAINING COMPLETE
==============================================================
  People registered : 5
  Total embeddings  : 14
  Skipped images    : 0
```

---

## 🏃 How to Run

```cmd
python main.py
```

### Controls While Running
| Key | Action |
|-----|--------|
| **`q`** | Quit the application and safely release the camera |
| **`r`** | Manually trigger OCR to read visible text aloud |

---

## 🛠️ Project Structure

```
├── main.py                  # Core application loop (YOLOv8 + routing)
├── train_faces.py           # Face embedding training pipeline (Facenet512)
├── requirements.txt         # Python dependencies
├── yolov8s.pt               # YOLOv8 Small model weights
├── yolov8n.pt               # YOLOv8 Nano model weights (backup)
├── known_faces/             # Face recognition image database
│   ├── PersonName.jpg       # Individual headshot photos
│   └── PersonName/          # Subfolder with multiple photos
│       └── *.jpg
└── modules/
    ├── audio_tts.py         # Async Windows SAPI voice engine (non-blocking)
    ├── face_recognizer.py   # Facenet512 face matching engine
    ├── ocr_reader.py        # EasyOCR text reading module
    ├── spatial_grid.py      # 3×3 grid positioning & distance estimation
    ├── road_signs.py        # Threaded road sign detection + OCR
    └── traffic_analyzer.py  # OpenCV HSV color analysis for traffic lights
```

---

## 🧠 Face Recognition Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    TRAINING (Offline)                     │
│                                                          │
│  known_faces/  ──►  extract_faces  ──►  Facenet512       │
│  (images)          (largest face)      (512-dim vector)  │
│                                            │             │
│                                            ▼             │
│                                   face_embeddings.pkl    │
│                                   (per-person DB)        │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                    RUNTIME (Live)                         │
│                                                          │
│  Webcam Frame  ──►  Detect Largest  ──►  Facenet512      │
│                     Face (OpenCV)       (512-dim vector)  │
│                                             │            │
│                                             ▼            │
│                                    Cosine Similarity     │
│                                    vs. trained DB        │
│                                             │            │
│                                             ▼            │
│                                 Match ≥ 0.55 ?           │
│                                   YES → TTS: "I see X"  │
│                                   NO  → (ignore)        │
└──────────────────────────────────────────────────────────┘
```

---

## 🚧 Future Roadmap

- **Indian Road Sign Detection**: Train a custom YOLOv8 model on an Indian Road Sign dataset to recognize iconography-based signs (which don't rely on text).
- **Raspberry Pi Deployment**: Optimize for edge deployment on Raspberry Pi with camera module.
- **Multi-Language OCR**: Extend EasyOCR to support Hindi and other regional languages.
- **Distance Estimation**: Use monocular depth estimation for more accurate obstacle distance measurement.

---

## 👥 Team

Built with ❤️ by the AMC Institutions team.
