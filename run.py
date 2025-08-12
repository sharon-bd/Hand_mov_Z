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
        # Ensure required modules are available
        import pygame
        print("✅ Pygame imported successfully")
        
        # Try to import other required modules
        try:
            import cv2
            print("✅ OpenCV imported successfully")
        except ImportError:
            print("⚠️ OpenCV not found, camera functions will be disabled")
        
        # Try to import the main menu system
        try:
            print("🔄 Loading main menu system...")
            from main import main as start_main_menu
            print("✅ Main menu system imported successfully")
            
            # Start the main menu
            print("🚀 Starting main menu...")
            start_main_menu()
            
            except ImportError as e:
                print(f"❌ Main menu import error: {e}")
                print("Trying to import GameLauncher directly...")
                try:
                from main import GameLauncher
                print("✅ GameLauncher imported successfully")
                
                # Run the game
                print("🚀 Launching game...")
                game_launcher = GameLauncher()
                game_launcher.run()
                
            except ImportError as e2:
                print(f"❌ GameLauncher import error: {e2}")
                # As last resort - try to run the game directly
                print("Trying to run game directly as last resort...")
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