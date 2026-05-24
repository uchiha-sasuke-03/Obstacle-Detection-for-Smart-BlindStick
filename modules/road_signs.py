import time
import threading

class RoadSignDetector:
    def __init__(self):
        self.last_spoken = 0
        self.cooldown = 10  # Don't auto-read signs constantly, wait 10 seconds between signs
        self.is_processing = False

    def analyze_sign(self, cropped_img, ocr_module, tts_manager):
        if self.is_processing or (time.time() - self.last_spoken < self.cooldown):
            return
            
        self.is_processing = True

        def _process():
            try:
                # In the future, this is where we would load a custom Indian Road Sign 
                # classification model (e.g. YOLOv8 trained on Indian signs).
                # For now, we will use our OCR module to read any text written on the sign!
                
                # Note: ocr_module.read_text automatically spawns a thread, 
                # but we will just call the inner logic if we want to wait, or just pass it directly.
                
                tts_manager.speak("Road sign detected, scanning text.")
                
                # We call the OCR on just the cropped sign image
                ocr_module.read_text(cropped_img, tts_manager)
                
                self.last_spoken = time.time()
            except Exception as e:
                pass
            finally:
                self.is_processing = False

        threading.Thread(target=_process, daemon=True).start()

road_sign_detector = RoadSignDetector()
