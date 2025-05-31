#!/usr/bin/env python
"""
Entry Point for Hand Gesture Car Control Game
"""

import sys
import os
import traceback

# Add the project directory to Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

def main():
    """Main entry point"""
    print("🎮 Starting Hand Gesture Car Control Game...")
    
    try:
        # וודא שהמודולים הנדרשים קיימים
        import pygame
        print("✅ Pygame imported successfully")
        
        # נסה לייבא את המודולים נדרשים אחרים
        try:
            import cv2
            print("✅ OpenCV imported successfully")
        except ImportError:
            print("⚠️ OpenCV not found, camera functions will be disabled")
        
        # נסה לייבא את המשחק
        try:
            from main_game import GameLauncher
            print("✅ Game launcher imported successfully")
            
            # הפעל את המשחק
            print("🚀 Launching game...")
            game_launcher = GameLauncher()
            game_launcher.run()
            
        except ImportError as e:
            print(f"❌ Game launcher import error: {e}")
            # נסה להריץ את המשחק ישירות
            from game.start_game import run_game
            run_game("normal")
            
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Please ensure all required modules are installed:")
        print("  pip install pygame opencv-python numpy")
        traceback.print_exc()
        return 1
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Critical error: {e}")
        traceback.print_exc()
        sys.exit(1)