import mediapipe as mp
try:
    print(f"MediaPipe version: {mp.__version__}")
    print(f"Solutions: {mp.solutions}")
    print("Success: mp.solutions exists")
except AttributeError as e:
    print(f"FAIL: {e}")
except Exception as e:
    print(f"Error: {e}")
