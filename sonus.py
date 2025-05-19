import pygame
import pyaudio
import numpy as np
import threading
import sys
import math

# =======================
# Audio Configuration
# =======================
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
audio_data_buffer = {"data": np.zeros(CHUNK, dtype=np.int16)}

def listen():
    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            audio_data_buffer["data"] = audio_data
        except Exception as e:
            print("Audio Error:", e)
            break

listener_thread = threading.Thread(target=listen)
listener_thread.daemon = True
listener_thread.start()

# =======================
# Pygame Setup
# =======================
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Sound Shape Visualizer")
clock = pygame.time.Clock()

BG_COLOR = (0, 0, 0)
SHAPE_COLOR = (0, 255, 255)

# =======================
#  Functions
# =======================
def get_audio_features(samples):
    # Volume
    volume = np.sqrt(np.mean(samples ** 2)) / 32768  # 0 to 1

    # Frequency
    fft = np.fft.fft(samples)
    freqs = np.fft.fftfreq(len(samples), 1 / RATE)
    fft = fft[:len(fft)//2]
    freqs = freqs[:len(freqs)//2]
    mag = np.abs(fft)
    dominant_freq = freqs[np.argmax(mag)]
    return volume, dominant_freq


# =======================
# Main Loop
# =======================
running = True
while running:
    clock.tick(60)
    screen.fill(BG_COLOR)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.draw.line(screen, (75, 0, 130), (0, HEIGHT // 2), (WIDTH, HEIGHT // 2), 5)

    pygame.draw.line(screen, (75, 0, 130), (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), 5)

    samples = audio_data_buffer["data"]
    volume, freq = get_audio_features(samples)
    pygame.display.flip()

# =======================
# Cleanup
# =======================
stream.stop_stream()
stream.close()
p.terminate()
pygame.quit()
sys.exit()
