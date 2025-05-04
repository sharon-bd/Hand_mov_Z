#!/usr/bin/env python
"""
Run Script for Hand Gesture Car Control System

This script provides a simple way to run the Hand Gesture Car Control System.
It checks for dependencies before launching the main application.
"""

import os
import sys
import subprocess
import platform
import pygame

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = ['pygame', 'mediapipe', 'opencv-python', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies(packages):
    """Install missing dependencies"""
    print("Installing missing dependencies...")
    
    # Use Python executable from the current environment
    python = sys.executable
    
    try:
        subprocess.check_call([python, '-m', 'pip', 'install'] + packages)
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("Failed to install dependencies. Please install them manually:")
        print("pip install " + " ".join(packages))
        return False

class Game:
    def __init__(self):
        self.running = True

    def setup(self):
        """Placeholder for setup operations"""
        pass

    def draw_menu(self):
        """Placeholder for drawing the menu"""
        pass

    def cleanup(self):
        """Placeholder for cleanup operations"""
        pass

    def run(self):
        """Main game loop for training mode"""
        self.setup()  # Call setup without checking return value
        
        while self.running:
            # ...existing code...
            if pygame.display.get_surface():
                self.draw_menu()
            else:
                print("Display surface is not available, exiting game loop")
                self.running = False
            # ...existing code...

def main():
    """Main function to run the application"""
    print("Hand Gesture Car Control System Launcher")
    print("========================================")
    
    # Check for Python version
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required.")
        return
    
    # Check for dependencies
    missing_packages = check_dependencies()
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        user_input = input("Do you want to install them now? (y/n): ")
        
        if user_input.lower() in ['y', 'yes']:
            if not install_dependencies(missing_packages):
                return
        else:
            print("Cannot continue without required dependencies.")
            return
    
    # Run the main application
    try:
        print("Starting the application...")
        
        # Get the path to main.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        main_script = os.path.join(current_dir, 'main.py')
        
        # Run the main script
        if platform.system() == 'Windows':
            os.system(f'"{sys.executable}" "{main_script}"')
        else:
            os.system(f'"{sys.executable}" "{main_script}"')
            
    except Exception as e:
        print(f"Error launching the application: {e}")

if __name__ == "__main__":
    main()