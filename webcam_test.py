from ultralytics import YOLO
import cv2

def main():
    print("Loading YOLOv8s model (a slightly larger, more accurate model)...")
    # Upgraded from 'n' (nano) to 's' (small) for better detection accuracy
    model = YOLO('yolov8s.pt')

    print("Opening webcam... (Press 'q' in the video window to quit)")
    # Lowered confidence from 0.5 to 0.4 to detect more objects
    model.predict(source="0", show=True, conf=0.4)

if __name__ == "__main__":
    main()
