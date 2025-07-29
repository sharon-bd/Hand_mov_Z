#!/usr/bin/env python3
"""
Sound System Test for Hand Gesture Car Game
Tests pygame mixer initialization and sound generation
"""

import pygame
import time
import sys

def test_pygame_mixer():
    """Test basic pygame mixer initialization"""
    print("🔍 Testing pygame mixer initialization...")
    
    try:
        pygame.init()
        print("✅ Pygame initialized")
    except Exception as e:
        print(f"❌ Pygame initialization failed: {e}")
        return False
    
    try:
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        print("✅ Pygame mixer initialized successfully")
        
        # Get mixer info
        freq, format_type, channels = pygame.mixer.get_init()
        print(f"   Frequency: {freq} Hz")
        print(f"   Format: {format_type}")
        print(f"   Channels: {channels}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pygame mixer initialization failed: {e}")
        return False

def test_numpy_availability():
    """Test if numpy is available for sound generation"""
    print("\n🔍 Testing NumPy availability...")
    
    try:
        import numpy as np
        print("✅ NumPy is available")
        print(f"   NumPy version: {np.__version__}")
        return True
    except ImportError:
        print("❌ NumPy is not available - will use simple sound fallback")
        return False

def test_simple_sound():
    """Test generating and playing a simple sound"""
    print("\n🔍 Testing simple sound generation...")
    
    try:
        # Generate a simple beep
        duration = 1.0  # 1 second
        sample_rate = 22050
        frequency = 440  # A note
        
        # Create simple sine wave manually
        import math
        samples = int(sample_rate * duration)
        wave_array = []
        
        for i in range(samples):
            time_point = float(i) / sample_rate
            value = int(0.3 * 32767 * math.sin(2 * math.pi * frequency * time_point))
            wave_array.append([value, value])  # Stereo
        
        sound = pygame.sndarray.make_sound(wave_array)
        print("✅ Simple sound generated successfully")
        
        # Play the sound
        print("🔊 Playing test sound...")
        sound.play()
        time.sleep(duration + 0.5)  # Wait for sound to finish
        
        return True
        
    except Exception as e:
        print(f"❌ Simple sound generation failed: {e}")
        return False

def test_numpy_sound():
    """Test generating sound with NumPy"""
    print("\n🔍 Testing NumPy sound generation...")
    
    try:
        import numpy as np
        
        # Generate engine-like sound
        duration = 1.0
        sample_rate = 22050
        samples = int(sample_rate * duration)
        
        # Create time array
        t = np.linspace(0, duration, samples, False)
        
        # Generate multi-harmonic sound
        base_freq = 150
        wave = np.zeros(samples)
        
        # Add harmonics
        wave += 0.4 * np.sin(2 * np.pi * base_freq * t)
        wave += 0.3 * np.sin(2 * np.pi * base_freq * 2 * t)
        wave += 0.2 * np.sin(2 * np.pi * base_freq * 3 * t)
        
        # Normalize and convert to 16-bit
        wave = np.clip(wave, -1, 1)
        wave = (wave * 0.5 * 32767).astype(np.int16)
        
        # Convert to stereo
        stereo_wave = np.array([wave, wave]).T
        
        sound = pygame.sndarray.make_sound(stereo_wave)
        print("✅ NumPy sound generated successfully")
        
        # Play the sound
        print("🔊 Playing NumPy test sound...")
        sound.play()
        time.sleep(duration + 0.5)  # Wait for sound to finish
        
        return True
        
    except Exception as e:
        print(f"❌ NumPy sound generation failed: {e}")
        return False

def test_sound_manager():
    """Test the game's SoundManager class"""
    print("\n🔍 Testing SoundManager from game...")
    
    try:
        # Import the SoundManager from the game
        sys.path.append(r'c:\Users\Sharon\JohnBryce-python\Project2\Hand_mov_Z\game')
        from start_game import SoundManager
        
        # Create sound manager
        sound_manager = SoundManager()
        
        if sound_manager.muted:
            print("⚠️ SoundManager initialized but is muted")
            return False
        
        print("✅ SoundManager initialized successfully")
        
        # Test engine sound generation
        print("🔊 Testing engine sound generation...")
        engine_sound = sound_manager.generate_engine_sound(0.5, 1.0)
        
        if engine_sound:
            print("✅ Engine sound generated")
            engine_sound.play()
            time.sleep(1.5)
        else:
            print("❌ Engine sound generation failed")
            return False
            
        # Test collision sound
        print("🔊 Testing collision sound...")
        sound_manager.play("collision")
        time.sleep(1.0)
        
        # Test boost sound
        print("🔊 Testing boost sound...")
        sound_manager.play("boost")
        time.sleep(1.0)
        
        # Cleanup
        sound_manager.cleanup()
        print("✅ SoundManager test completed")
        
        return True
        
    except Exception as e:
        print(f"❌ SoundManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all sound system tests"""
    print("🎵 Sound System Diagnostic Tool")
    print("=" * 50)
    
    tests = [
        ("Pygame Mixer", test_pygame_mixer),
        ("NumPy Availability", test_numpy_availability),
        ("Simple Sound", test_simple_sound),
        ("NumPy Sound", test_numpy_sound),
        ("SoundManager", test_sound_manager)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("🎵 SOUND SYSTEM TEST RESULTS")
    print("=" * 50)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} : {status}")
    
    # Recommendations
    print("\n🔧 RECOMMENDATIONS:")
    
    if not results.get("Pygame Mixer", False):
        print("❌ Pygame mixer failed - check audio drivers and sound system")
    
    if not results.get("NumPy Availability", False):
        print("⚠️ NumPy not available - install with: pip install numpy")
    
    if not results.get("Simple Sound", False):
        print("❌ Basic sound failed - check system audio settings")
    
    if not results.get("SoundManager", False):
        print("❌ Game sound system failed - check game code and dependencies")
    
    if all(results.values()):
        print("✅ All tests passed - sound system should work in game!")
    else:
        print("⚠️ Some tests failed - sound issues expected in game")
    
    print("\nPress Enter to exit...")
    input()

if __name__ == "__main__":
    main()
