import pyttsx3
import threading

def _run():
    try:
        import pythoncom
        pythoncom.CoInitialize()
        print("CoInitialize successful")
    except Exception as e:
        print("CoInitialize failed:", e)

    try:
        e = pyttsx3.init()
        print("Engine initialized")
        e.say('test thread')
        e.runAndWait()
        print('Speech done')
    except Exception as e:
        print("Speech failed:", e)

print("Starting thread")
t = threading.Thread(target=_run)
t.start()
t.join()
print("Main thread done")
