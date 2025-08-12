#!/usr/bin/env python3
"""
Simple sound test to debug the sound system
"""

import sys
import os
import pygame
import time
import numpy as np
import math

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_basic_pygame_sound():
    """Test basic pygame sound functionality"""
    print("🔍 Testing basic pygame sound...")
    
    try:
        # Initialize pygame
        pygame.init()
        
        # Initialize mixer with better settings
        pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=1024)
        pygame.mixer.init()
        
        print("✅ Pygame and mixer initialized successfully")
        print(f"   Mixer settings: {pygame.mixer.get_init()}")
        
        # Test generating and playing a simple sound
        print("🔊 Generating test sound...")
        
        duration = 2.0
        sample_rate = 22050
        frequency = 440  # A note
        samples = int(sample_rate * duration)
        
        # Create sine wave using numpy
        wave_array = np.zeros((samples, 2), dtype=np.int16)
        for i in range(samples):
            time_point = float(i) / sample_rate
            amplitude = 0.3
            value = int(amplitude * 32767 * math.sin(2 * math.pi * frequency * time_point))
            wave_array[i] = [value, value]  # Stereo
        
        # Create pygame sound
        sound = pygame.sndarray.make_sound(wave_array)
        sound.set_volume(0.7)
        
        print("✅ Test sound created successfully")
        print("🔊 Playing test sound for 2 seconds...")
        
        # Play sound
        channel = sound.play()
        
        # Wait for sound to finish
        start_time = time.time()
        while channel.get_busy() and (time.time() - start_time < 3):
            time.sleep(0.1)
        
        print("✅ Test sound completed!")
        
        pygame.quit()
        return True
        
    except Exception as e:
        print(f"❌ Basic sound test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_game_sound_manager():
    """Test the game's sound manager"""
    print("\n🔍 Testing game sound manager...")
    
    try:
        # Import from the game folder
        game_path = os.path.join(current_dir, 'game')
        sys.path.insert(0, game_path)
        from game.start_game import SoundManager
        
        print("✅ Successfully imported SoundManager")
        
        # Create sound manager
        sound_manager = SoundManager()
        
        if sound_manager.muted:
            print("⚠️ Sound manager is muted!")
            return False
        
        print("✅ Sound manager created successfully")
        
        # Test engine sound
        print("🔊 Testing engine sound generation...")
        engine_sound = sound_manager.generate_engine_sound(0.5, 1.0)
        
        if engine_sound:
            print("✅ Engine sound generated successfully")
            print("🔊 Playing engine sound...")
            channel = engine_sound.play()
            
            # Wait for sound
            start_time = time.time()
            while channel and channel.get_busy() and (time.time() - start_time < 2):
                time.sleep(0.1)
            
            print("✅ Engine sound played successfully")
        else:
            print("❌ Failed to generate engine sound")
            return False
        
        # Test collision sound
        print("🔊 Testing collision sound...")
        sound_manager.play("collision")
        time.sleep(1)
        
        # Test boost sound
        print("🔊 Testing boost sound...")
        sound_manager.play("boost")
        time.sleep(1)
        
        # Cleanup
        sound_manager.cleanup()
        print("✅ Sound manager test completed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Game sound manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run sound diagnostic tests"""
    print("🎵 SOUND SYSTEM DIAGNOSTIC")
    print("=" * 50)
    
    # Test 1: Basic pygame sound
    basic_test = test_basic_pygame_sound()
    
    # Test 2: Game sound manager
    game_test = test_game_sound_manager()
    
    # Results
    print("\n" + "=" * 50)
    print("🎵 TEST RESULTS")
    print("=" * 50)
    print(f"Basic Pygame Sound: {'✅ PASS' if basic_test else '❌ FAIL'}")
    print(f"Game Sound Manager: {'✅ PASS' if game_test else '❌ FAIL'}")
    
    if basic_test and game_test:
        print("\n✅ All tests passed! Sound system should work in the game.")
        print("   Try running the game and press 'M' to unmute if needed.")
    elif basic_test:
        print("\n⚠️ Basic sound works, but game sound manager has issues.")
        print("   Check the SoundManager class implementation.")
    else:
        print("\n❌ Basic sound system is not working.")
        print("   Check your audio drivers and system volume.")
    
    print("\nPress Enter to exit...")
    input()

if __name__ == "__main__":
    main()
