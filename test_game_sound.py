#!/usr/bin/env python3
"""
Quick test to verify sound system works in the game
"""

import sys
import os

# Add the game directory to Python path
game_dir = r'c:\Users\Sharon\JohnBryce-python\Project2\Hand_mov_Z\game'
sys.path.insert(0, game_dir)

# Change to game directory
os.chdir(r'c:\Users\Sharon\JohnBryce-python\Project2\Hand_mov_Z')

try:
    print("🎵 Testing game sound system...")
    
    # Import the SoundManager class
    from start_game import SoundManager
    
    # Create sound manager
    print("\n🔧 Creating SoundManager...")
    sound_manager = SoundManager()
    
    if sound_manager.muted:
        print("❌ SoundManager is muted - check initialization errors above")
    else:
        print("✅ SoundManager created successfully")
        
        # Test collision sound
        print("\n🔊 Testing collision sound...")
        sound_manager.play("collision")
        
        # Test boost sound  
        print("\n🔊 Testing boost sound...")
        sound_manager.play("boost")
        
        # Test engine sound
        print("\n🔊 Testing engine sound...")
        engine_sound = sound_manager.generate_engine_sound(0.5)
        if engine_sound:
            print("✅ Engine sound generated")
            channel = engine_sound.play()
            if channel:
                print("✅ Engine sound played")
            else:
                print("❌ Engine sound failed to play")
        else:
            print("❌ Engine sound generation failed")
        
        # Clean up
        sound_manager.cleanup()
        print("\n✅ Sound test completed")

except Exception as e:
    print(f"❌ Error testing sound system: {e}")
    import traceback
    traceback.print_exc()

print("\nPress Enter to exit...")
input()
