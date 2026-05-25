import cv2
import time
from ultralytics import YOLO

# Import our custom modules
from modules.audio_tts import tts
from modules.spatial_grid import analyze_position
from modules.ocr_reader import ocr
from modules.face_recognizer import face_rec
from modules.traffic_analyzer import traffic_analyzer
from modules.road_signs import road_sign_detector

def main():
    print("Initializing Advanced Blind Stick System...")
    tts.speak("System starting up")
    
    model = YOLO('yolov8s.pt')
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # To avoid speaking the same object constantly
    last_spoken_objects = {}
    
    frame_count = 0

    print("System ready. Press 'q' to quit, 'r' to read text.")
    tts.speak("System ready.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_height, frame_width = frame.shape[:2]
        frame_count += 1

        # 1. Run YOLO Object Detection (every frame)
        results = model.predict(frame, verbose=False, conf=0.5)
        
        current_objects = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get class name
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                
                # Get coordinates
                xyxy = box.xyxy[0].cpu().numpy()
                
                # 2. Grid & Spatial Awareness
                location, distance = analyze_position(xyxy, frame_width, frame_height)
                
                object_id = f"{class_name}_{location}_{distance}"
                current_objects.append(object_id)
                
                # For regular objects, only speak aloud if they are "close"
                if class_name not in ["traffic light", "stop sign"]:
                    if distance == "close":
                        if object_id not in last_spoken_objects or (time.time() - last_spoken_objects[object_id]) > 3:
                            tts.speak(f"{class_name} {distance}, {location}")
                            last_spoken_objects[object_id] = time.time()
                else:
                    # For traffic lights and signs, we pass them constantly and let their own internal modules 
                    # decide when they have a clear enough image to speak (using their own cooldowns).
                    if class_name == "traffic light":
                        cropped = frame[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                        if cropped.size > 0:
                            traffic_analyzer.analyze_and_speak(cropped, tts)
                    elif class_name == "stop sign":
                        cropped = frame[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                        if cropped.size > 0:
                            road_sign_detector.analyze_sign(cropped, ocr, tts)

                # Draw bounding box for visual debugging
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} {distance} {location}", (int(xyxy[0]), int(xyxy[1])-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Clean up old spoken objects memory
        current_time = time.time()
        keys_to_delete = [k for k, v in last_spoken_objects.items() if (current_time - v) > 10]
        for k in keys_to_delete:
            del last_spoken_objects[k]

        # 3. Face Recognition (run every 10 frames to save CPU)
        if frame_count % 10 == 0:
            face_rec.recognize(frame, tts)

        # Draw UI
        cv2.putText(frame, "Press 'r' to read text, 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Blind Stick View", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # 4. OCR Trigger
        elif key == ord('r'):
            tts.speak("Scanning for text")
            ocr.read_text(frame, tts)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    tts.stop()

if __name__ == "__main__":
    main()
