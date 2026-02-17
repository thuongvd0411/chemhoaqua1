import threading
import time

try:
    import speech_recognition as sr
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    print("Voice Control disabled: SpeechRecognition/PyAudio not found.")

# Shared state for commands
class VoiceController:
    _instance = None
    _command_queue = []
    _running = False
    _thread = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = VoiceController()
        return cls._instance

    def __init__(self):
        self._command_queue = []
        if not VOICE_AVAILABLE:
            return

        self.recognizer = sr.Recognizer()
        try:
            self.microphone = sr.Microphone()
            # Calibrate ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            self._running = True
            self._thread = threading.Thread(target=self._listen_loop, daemon=True)
            self._thread.start()
        except Exception as e:
            print(f"Voice Init Error: {e}")
            self._running = False

    def _listen_loop(self):
        if not VOICE_AVAILABLE: return
        print("Voice Listener Started...")
        while self._running:
            try:
                with self.microphone as source:
                    # Listen for audio (short timeout to keep loop responsive)
                    try:
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                    except sr.WaitTimeoutError:
                        continue

                    try:
                        # Recognize Vietnamese
                        text = self.recognizer.recognize_google(audio, language="vi-VN")
                        print(f"Recognized: {text}")
                        self._process_command(text.lower())
                    except sr.UnknownValueError:
                        pass # ensure loop continues
                    except sr.RequestError:
                        print("Google API unavailable")
                        time.sleep(5)
            except Exception as e:
                print(f"Voice Error: {e}")
                time.sleep(1)

    def _process_command(self, text):
        # Map phrases to commands
        if "bắt đầu" in text or "chơi" in text or "vô" in text:
            self.add_command("START")
        elif "tiếp" in text or "màn tiếp" in text or "kế tiếp" in text or "qua màn" in text:
            self.add_command("NEXT")
        elif "lại" in text or "thử lại" in text or "chơi lại" in text:
            self.add_command("RETRY")
            
    def add_command(self, cmd):
        self._command_queue.append(cmd)

    def get_last_command(self):
        if self._command_queue:
            return self._command_queue.pop(0)
        return None

# Global helper to start
def start_voice_listener():
    return VoiceController.get_instance()
