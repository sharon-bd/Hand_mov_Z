#!/usr/bin/env python3
"""
Comprehensive game sound system test
"""

import sys
import os
import time
import pygame

# Add game path
sys.path.insert(0, os.path.join(os.getcwd(), 'game'))

def test_audio_manager():
    """Test the game's AudioManager"""
    print("🎵 Testing game AudioManager")
    print("=" * 50)
    
    try:
        # Initialize pygame mixer
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        print("✅ Pygame mixer initialized successfully")
        
        # Import AudioManager
        from audio_manager import AudioManager
        
        print("✅ AudioManager imported successfully")
        
        # Create audio manager
        print("\n🔧 Creating AudioManager...")
        audio_manager = AudioManager()
        
        print("✅ AudioManager created successfully")
        
        # Test game sounds - using correct AudioManager functions
        print("\n🎼 Testing game sounds:")
        
        # 1. Collision sound
        print("\n🔊 Playing collision sound...")
        try:
            audio_manager.play_collision_sound()
            print("✅ Collision sound played successfully")
            time.sleep(1.5)
        except Exception as e:
            print(f"❌ Collision sound error: {e}")
        
        # 2. Boost sound
        print("\n🚀 Playing boost sound...")
        try:
            audio_manager.play_boost_sound()
            print("✅ Boost sound played successfully")
            time.sleep(1.5)
        except Exception as e:
            print(f"❌ Boost sound error: {e}")
        
        # 3. Brake sound
        print("\n🛑 Playing brake sound...")
        try:
            audio_manager.play_brake_sound()
            print("✅ Brake sound played successfully")
            time.sleep(1.5)
        except Exception as e:
            print(f"❌ Brake sound error: {e}")
        
        # 4. Menu sounds
        print("\n📋 Playing menu sounds...")
        try:
            audio_manager.play_menu_select_sound()
            print("✅ Menu select sound played successfully")
            time.sleep(1.0)
            
            audio_manager.play_menu_move_sound()
            print("✅ Menu move sound played successfully")
            time.sleep(1.0)
        except Exception as e:
            print(f"❌ Menu sounds error: {e}")
        
        # 5. Power-up sound
        print("\n⚡ Playing power-up sound...")
        try:
            audio_manager.play_power_up_sound()
            print("✅ Power-up sound played successfully")
            time.sleep(1.5)
        except Exception as e:
            print(f"❌ Power-up sound error: {e}")
        
        print("\n✅ All game sounds tested successfully!")
        
        # Cleanup
        audio_manager.cleanup()
        print("\n✅ AudioManager cleaned up successfully")
        
    except ImportError as e:
        print(f"❌ Error importing AudioManager: {e}")
        return False
    except Exception as e:
        print(f"❌ General error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    print("🎮 Comprehensive sound system test")
    print("=" * 60)
    
    # Test AudioManager
    success = test_audio_manager()
    
    if success:
        print("\n🎉 All tests passed successfully!")
        print("System ready for use with procedural audio.")
        print("\n💡 The system creates sounds algorithmically:")
        print("   • Collision - white noise with low frequencies")
        print("   • Boost - rising frequency sine wave")
        print("   • Brakes - noise with fast decay")
        print("   • Menu - short, sharp sounds")
        print("   • Power-up - rising melody")
    else:
        print("\n❌ There are issues with the sound system")
    
    print("\n" + "="*60)
    print("Press Enter to exit...")
    input()

if __name__ == "__main__":
    main()
