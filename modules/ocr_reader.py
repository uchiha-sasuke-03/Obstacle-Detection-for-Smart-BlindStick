import easyocr
import threading

class OCRReader:
    def __init__(self):
        self.reader = None
        self.is_reading = False
        self.last_text = ""

    def read_text(self, frame, tts_manager):
        if self.reader is None:
            tts_manager.speak("Please wait, loading text reading AI into memory for the first time.")
            self.reader = easyocr.Reader(['en'], gpu=False)
            
        if self.is_reading:
            return
        """
        Reads text from the frame and speaks it out loud.
        This runs in a thread to prevent blocking the video feed.
        """
        if self.is_reading:
            return # Don't start a new reading if one is already in progress

        def _process():
            self.is_reading = True
            try:
                # readtext returns a list of tuples: (bbox, text, confidence)
                results = self.reader.readtext(frame)
                detected_text = " ".join([res[1] for res in results if res[2] > 0.4])
                
                if detected_text and detected_text != self.last_text:
                    tts_manager.speak(f"Text says: {detected_text}")
                    self.last_text = detected_text
            except Exception as e:
                print(f"OCR Error: {e}")
            finally:
                self.is_reading = False

        # Run OCR in a background thread
        threading.Thread(target=_process, daemon=True).start()

ocr = OCRReader()
