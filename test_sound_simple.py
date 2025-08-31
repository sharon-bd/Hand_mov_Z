#!/usr/bin/env python3
"""
Simple sound system test - plays procedural audio
"""

import pygame
import time
import numpy as np

def generate_test_sound(frequency, duration=1.0, sample_rate=44100):
    """Creates a sound at a given frequency"""
    frames = int(duration * sample_rate)
    arr = np.zeros((frames, 2), dtype=np.int16)
    
    # Create sine wave
    for i in range(frames):
        wave = 4096 * np.sin(2 * np.pi * frequency * i / sample_rate)
        arr[i][0] = int(wave)  # Left channel
        arr[i][1] = int(wave)  # Right channel
    
    return pygame.sndarray.make_sound(arr)

def main():
    print("🎵 Testing procedural sound system")
    print("=" * 40)
    
    # Initialize pygame
    pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
    pygame.mixer.init()
    
    print("✅ pygame mixer initialized successfully")
    
    # Testing different sounds
    test_sounds = [
        ("Low engine sound", 120, 2.0),
        ("Boost sound", 400, 1.5),
        ("Collision sound", 200, 1.0),
        ("Menu sound", 800, 0.5),
    ]
    
    for name, freq, duration in test_sounds:
        print(f"\n🔊 Playing: {name} (frequency: {freq} Hz)")
        
        try:
            # Create the sound
            sound = generate_test_sound(freq, duration)
            
            # Play the sound
            channel = sound.play()
            
            if channel:
                print(f"✅ Sound playing - waiting {duration} seconds...")
                
                # Wait until sound finishes
                while channel.get_busy():
                    time.sleep(0.1)
                    
                print("✅ Sound finished")
            else:
                print("❌ Error playing sound")
                
        except Exception as e:
            print(f"❌ Error creating sound: {e}")
    
    print("\n🎉 Sound test completed!")
    print("If you heard sounds - the system works perfectly!")
    
    # Cleanup
    pygame.mixer.quit()
    
    print("\nPress Enter to exit...")
    input()

if __name__ == "__main__":
    main()
