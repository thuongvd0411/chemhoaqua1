import wave
import math
import struct
import os
import random

ASSETS_DIR = "assets"
if not os.path.exists(ASSETS_DIR):
    os.makedirs(ASSETS_DIR)

def save_wave(filename, data, sample_rate=44100):
    with wave.open(filename, 'w') as obj:
        obj.setnchannels(1) # Mono
        obj.setsampwidth(2) # 16-bit
        obj.setframerate(sample_rate)
        
        # Convert float -1..1 to int16
        packed_data = bytearray()
        for sample in data:
            s_int = int(max(-1, min(1, sample)) * 32767)
            packed_data.extend(struct.pack('<h', s_int))
            
        obj.writeframes(packed_data)
    print(f"Saved {filename}")

def generate_tone(freq, duration, sample_rate=44100, volume=0.5):
    n_samples = int(sample_rate * duration)
    data = []
    for i in range(n_samples):
        t = float(i) / sample_rate
        # Sine wave with exponential decay
        val = math.sin(2 * math.pi * freq * t) * math.exp(-t * 5) * volume
        data.append(val)
    return data

def generate_noise(duration, sample_rate=44100, volume=0.5):
    n_samples = int(sample_rate * duration)
    data = []
    for i in range(n_samples):
        t = float(i) / sample_rate
        val = (random.random() * 2 - 1) * math.exp(-t * 3) * volume
        data.append(val)
    return data

if __name__ == "__main__":
    # Slice: High pitch beep
    slice_data = generate_tone(800, 0.1)
    # Add some noise
    noise = generate_noise(0.1, volume=0.2)
    for i in range(len(slice_data)):
        slice_data[i] += noise[i]
    save_wave(os.path.join(ASSETS_DIR, "slice.wav"), slice_data)
    
    # Bomb: Low noise
    bomb_data = generate_noise(0.5, volume=0.8)
    save_wave(os.path.join(ASSETS_DIR, "bomb.wav"), bomb_data)
    
    # Combo: Chord
    combo_data = []
    n_samples = int(44100 * 0.4)
    for i in range(n_samples):
        t = float(i) / 44100
        val = (math.sin(2 * math.pi * 400 * t) + 
               math.sin(2 * math.pi * 600 * t) + 
               math.sin(2 * math.pi * 800 * t)) * 0.3 * math.exp(-t*2)
        combo_data.append(val)
    save_wave(os.path.join(ASSETS_DIR, "combo.wav"), combo_data)

    # Game over
    game_over_data = generate_tone(150, 1.0)
    save_wave(os.path.join(ASSETS_DIR, "game_over.wav"), game_over_data)
