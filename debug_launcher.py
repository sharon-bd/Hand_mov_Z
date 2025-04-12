#!/usr/bin/env python
"""
Debug Launcher - A simple script to check the project structure and run the game directly
"""

import os
import sys
import importlib

def check_project_structure():
    """Check the project directory structure and report findings"""
    print("Examining project structure...")
    current_dir = os.path.dirname(os.path.abspath(__file__)) or "."
    print(f"Current directory: {current_dir}")
    
    # List files in current directory
    print("\nFiles in current directory:")
    files = os.listdir(current_dir)
    for file in sorted(files):
        print(f"  {file}")
    
    # Check for important files
    key_files = ['main.py', 'main_game.py', 'training_mode.py', 'config.py']
    for file in key_files:
        if file in files:
            print(f"✓ Found {file}")
        else:
            print(f"✗ Missing {file}")
    
    # Check for directories
    key_dirs = ['hand_detector', 'game']
    for dir_name in key_dirs:
        if os.path.isdir(os.path.join(current_dir, dir_name)):
            print(f"✓ Found directory: {dir_name}")
            # Check files inside directory
            dir_files = os.listdir(os.path.join(current_dir, dir_name))
            print(f"  Files in {dir_name}/: {', '.join(dir_files[:5])}{'...' if len(dir_files) > 5 else ''}")
        else:
            print(f"✗ Missing directory: {dir_name}")

def run_main_directly():
    """Try to run main.py directly"""
    print("\nAttempting to run main.py directly...")
    
    try:
        # First check if main.py exists
        if os.path.exists("main.py"):
            # Try to import main module
            print("Importing main module...")
            main_module = importlib.import_module("main")
            
            # Try to call main function if it exists
            if hasattr(main_module, "main"):
                print("Calling main.main()...")
                main_module.main()
            else:
                print("The main module doesn't have a main() function.")
                # Try running GameLauncher directly if it exists
                if hasattr(main_module, "GameLauncher"):
                    print("Found GameLauncher class, creating instance...")
                    game = main_module.GameLauncher()
                    print("Running game...")
                    game.run()
        else:
            print("main.py file not found in current directory!")
            
            # Look for other main files
            if os.path.exists("main_game.py"):
                print("Found main_game.py, trying to run it...")
                main_game = importlib.import_module("main_game")
                if hasattr(main_game, "main"):
                    main_game.main()
                else:
                    print("No main() function in main_game.py")
            
            elif os.path.exists("training_mode.py"):
                print("Found training_mode.py, trying to run it...")
                training = importlib.import_module("training_mode")
                if hasattr(training, "main"):
                    training.main()
                else:
                    print("No main() function in training_mode.py")
            
    except Exception as e:
        print(f"Error running main: {e}")
        import traceback
        traceback.print_exc()

def try_alternative_launch():
    """Try to launch game using alternative methods"""
    print("\nTrying alternative launch methods...")
    
    try:
        # Method 1: Try importing and running training mode directly
        if os.path.exists("training_mode.py"):
            print("Launching training mode directly...")
            from training_mode import main as training_main
            training_main()
            return True
    except Exception as e:
        print(f"Training mode launch failed: {e}")
    
    try:
        # Method 2: Try importing and running GameLauncher directly
        if os.path.exists("main_game.py"):
            print("Launching main_game directly...")
            from main_game import GameLauncher
            game = GameLauncher()
            game.run()
            return True
    except Exception as e:
        print(f"Main game launch failed: {e}")
    
    try:
        # Method 3: Try using training mode module directly
        if os.path.exists("training_mode.py"):
            print("Creating training mode instance directly...")
            from training_mode import TrainingMode
            training = TrainingMode()
            training.run()
            return True
    except Exception as e:
        print(f"Training mode instance failed: {e}")
    
    return False

def create_minimal_launcher():
    """Create a minimal launcher for the game"""
    print("\nCreating a minimal launcher...")
    
    try:
        launcher_code = """#!/usr/bin/env python
import pygame
import sys
import os

# Try to import the game components
try:
    from training_mode import TrainingMode
    
    def main():
        print("Starting Hand Gesture Car Control System...")
        training = TrainingMode()
        try:
            training.run()
        finally:
            training.cleanup()
            print("Game closed.")
    
    if __name__ == "__main__":
        main()
except Exception as e:
    print(f"Error importing game modules: {e}")
    import traceback
    traceback.print_exc()
"""
        
        # Write to a new file
        with open("minimal_launcher.py", "w") as f:
            f.write(launcher_code)
        
        print("Created minimal_launcher.py. Run it with: python minimal_launcher.py")
        
    except Exception as e:
        print(f"Error creating minimal launcher: {e}")

def main():
    """Main function"""
    print("Hand Gesture Car Control - Debug Launcher")
    print("=========================================")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check project structure
    check_project_structure()
    
    # Try to run main directly
    run_main_directly()
    
    # If that failed, try alternative launch methods
    if not try_alternative_launch():
        # Create a minimal launcher as last resort
        create_minimal_launcher()
    
    print("\nDebug complete.")

if __name__ == "__main__":
    main()