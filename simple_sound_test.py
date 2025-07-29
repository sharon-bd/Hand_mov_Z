import pygame
import time

# Initialize pygame
pygame.init()

# Initialize mixer with different settings
try:
    pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=1024)
    pygame.mixer.init()
    print("✅ Pygame mixer initialized")
    
    # Get mixer info
    freq, format_type, channels = pygame.mixer.get_init()
    print(f"Mixer settings: {freq}Hz, format={format_type}, channels={channels}")
    
except Exception as e:
    print(f"❌ Mixer init failed: {e}")
    exit()

# Test with a simple tone
print("🔊 Testing simple tone...")

try:
    import math
    
    # Create a simple beep
    duration = 2.0
    sample_rate = 22050
    frequency = 440
    samples = int(sample_rate * duration)
    
    wave_array = []
    for i in range(samples):
        time_point = float(i) / sample_rate
        # Create sine wave
        amplitude = 0.2  # Lower volume
        value = int(amplitude * 32767 * math.sin(2 * math.pi * frequency * time_point))
        wave_array.append([value, value])  # Stereo
    
    # Create sound
    sound = pygame.sndarray.make_sound(wave_array)
    print("✅ Sound created")
    
    # Play sound
    print("Playing sound for 2 seconds...")
    channel = sound.play()
    
    # Wait for sound to finish
    while channel.get_busy():
        time.sleep(0.1)
    
    print("✅ Sound playback completed")
    
except Exception as e:
    print(f"❌ Sound test failed: {e}")
    import traceback
    traceback.print_exc()

pygame.quit()
print("Test completed.")
