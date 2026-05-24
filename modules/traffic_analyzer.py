import cv2
import numpy as np
import threading
import time

class TrafficLightAnalyzer:
    def __init__(self):
        self.last_spoken = 0
        self.cooldown = 5  # Don't repeat the same light color for 5 seconds
        self.is_processing = False

    def analyze_and_speak(self, cropped_img, tts_manager):
        if self.is_processing or (time.time() - self.last_spoken < self.cooldown):
            return
            
        self.is_processing = True

        def _process():
            try:
                # Convert to HSV color space for easier color isolation
                hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

                # Red has two ranges in HSV (wrap around)
                # Lowering Saturation and Value to 50 to catch washed-out screens
                lower_red1 = np.array([0, 50, 50])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([160, 50, 50])
                upper_red2 = np.array([180, 255, 255])

                mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
                mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
                mask_red = cv2.bitwise_or(mask_red1, mask_red2)

                # Yellow
                lower_yellow = np.array([15, 50, 50])
                upper_yellow = np.array([35, 255, 255])
                mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

                # Green
                lower_green = np.array([40, 50, 50])
                upper_green = np.array([90, 255, 255])
                mask_green = cv2.inRange(hsv, lower_green, upper_green)

                # Count pixels
                red_pixels = cv2.countNonZero(mask_red)
                yellow_pixels = cv2.countNonZero(mask_yellow)
                green_pixels = cv2.countNonZero(mask_green)

                # Determine dominant color
                max_pixels = max(red_pixels, yellow_pixels, green_pixels)
                
                # Require at least some minimal threshold to avoid noise
                if max_pixels > 5:
                    if max_pixels == red_pixels:
                        tts_manager.speak("Traffic light is Red")
                    elif max_pixels == yellow_pixels:
                        tts_manager.speak("Traffic light is Yellow")
                    elif max_pixels == green_pixels:
                        tts_manager.speak("Traffic light is Green")
                    
                    self.last_spoken = time.time()
            except Exception as e:
                pass
            finally:
                self.is_processing = False

        threading.Thread(target=_process, daemon=True).start()

traffic_analyzer = TrafficLightAnalyzer()
