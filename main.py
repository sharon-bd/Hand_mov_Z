#!/usr/bin/env python
"""
Hand Gesture Car Control System - Main Entry Point

This is the main entry point for the Hand Gesture Car Control System.
It launches the application and allows users to choose between different game modes.
"""

import os
import sys
from main_game import GameLauncher

def main():
    """
    Main function that starts the application
    """
    print("Starting Hand Gesture Car Control System...")
    
    # Check if required packages are installed
    try:
        import pygame
        import mediapipe
        import cv2
        import numpy
    except ImportError as e:
        print(f"Error: Missing required package - {e}")
        print("Please install all required packages by running:")
        print("pip install -r requirements.txt")
        return
    
    # Launch the game
    game = GameLauncher()
    try:
        game.run()
    except Exception as e:
        print(f"Error in game: {e}")
        import traceback
        traceback.print_exc()
    finally:
        game.cleanup()
        print("Game closed.")

if __name__ == "__main__":
    main()