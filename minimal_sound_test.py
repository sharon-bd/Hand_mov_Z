import pygame
import numpy as np
import time

# Initialize pygame
pygame.init()
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)

print("Testing basic sound...")

try:
    # Create a simple beep
    duration = 1.0
    sample_rate = 22050
    frequency = 440
    samples = int(sample_rate * duration)
    
    # Generate sine wave as numpy array
    t = np.linspace(0, duration, samples, False)
    wave = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    # Convert to 16-bit stereo
    wave = (wave * 32767).astype(np.int16)
    stereo_wave = np.column_stack([wave, wave])
    stereo_wave = np.ascontiguousarray(stereo_wave)
    
    # Create and play sound
    sound = pygame.sndarray.make_sound(stereo_wave)
    print("Sound created, playing...")
    
    channel = sound.play()
    time.sleep(1.5)
    
    print("Sound test completed!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

pygame.quit()
