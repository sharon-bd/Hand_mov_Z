#!/usr/bin/env python
"""
Entry Point for Hand Gesture Car Control Game
"""

import sys
import os

# Add the project directory to Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

def main():
    """Main entry point"""
    print("🎮 Starting Hand Gesture Car Control Game...")
    
    try:
        from main_game import GameLauncher
        
        game_launcher = GameLauncher()
        game_launcher.run()
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Please ensure all required modules are installed.")
        return 1
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)