import queue
import threading
import time

class TTSManager:
    def __init__(self):
        self.speech_queue = queue.Queue()
        self.thread = None

    def _start_thread_if_needed(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._process_queue, daemon=True)
            self.thread.start()

    def _process_queue(self):
        # On Windows, COM objects must be initialized in the thread they are used
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except Exception:
            pass
            
        try:
            import win32com.client
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            # Rate ranges from -10 to 10
            speaker.Rate = 1
        except Exception as e:
            print("Failed to initialize SAPI:", e)
            speaker = None
        
        while True:
            text = self.speech_queue.get()
            if text is None:
                break
                
            if speaker is not None:
                try:
                    speaker.Speak(text)
                except Exception as e:
                    print("Speech failed:", e)
                    
            self.speech_queue.task_done()
            time.sleep(0.1) # Slight pause between speech

    def speak(self, text):
        self._start_thread_if_needed()
        # Only add to queue if the queue is empty to avoid huge backlogs of speech
        if self.speech_queue.empty():
            self.speech_queue.put(text)

    def stop(self):
        self.speech_queue.put(None)
        if self.thread is not None:
            self.thread.join()

tts = TTSManager()
