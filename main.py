#!/usr/bin/env python
"""
Hand Gesture Car Control System - Main Entry Point

This is the main entry point for the Hand Gesture Car Control System.
It launches the application and allows users to choose between different game modes.
"""

import os
import sys
import time
import pygame
import cv2
from main_game import GameLauncher

class TrainingMode:
    def __init__(self):
        self.running = True
        self.last_frame_time = time.time()
        self.fps_game = 0
        self.start_time = time.time()
        self.game_time = 0
        self.show_data_panel = False
        self.show_camera_feed = False
        self.data_panel_cv_window = "Data Panel"
        self.camera_window_name = None
        self.camera_thread = None
        self.last_gesture = None
        self.gesture_start_time = time.time()
        self.gesture_duration = 0

    def setup(self):
        """Setup the training mode"""
        pass

    def run(self):
        """Main game loop for training mode"""
        self.setup()
        
        while self.running:
            # Calculate time delta and FPS
            current_time = time.time()
            dt = current_time - self.last_frame_time
            self.last_frame_time = current_time
            
            # Safety check to prevent huge dt values
            if dt > 0.1:
                dt = 0.1
                
            # Calculate FPS - smoothed for display
            if dt > 0:
                instantaneous_fps = 1.0 / dt
                self.fps_game = (
                    0.9 * self.fps_game + 0.1 * instantaneous_fps
                    if self.fps_game > 0 else instantaneous_fps
                )
            
            # Update game time
            self.game_time = time.time() - self.start_time
            
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    # Toggle data panel
                    elif event.key == pygame.K_d:
                        self.show_data_panel = not self.show_data_panel
                        if self.show_data_panel:
                            try:
                                cv2.namedWindow(self.data_panel_cv_window, cv2.WINDOW_NORMAL)
                                cv2.resizeWindow(self.data_panel_cv_window, 600, 500)
                            except Exception as e:
                                print(f"Error creating data panel window: {e}")
                        else:
                            try:
                                cv2.destroyWindow(self.data_panel_cv_window)
                            except:
                                pass
                    # Toggle camera feed in separate window
                    elif event.key == pygame.K_v:
                        self.show_camera_feed = not self.show_camera_feed
                        if not self.show_camera_feed and self.camera_window_name:
                            try:
                                cv2.destroyWindow(self.camera_window_name)
                            except:
                                pass
                        elif self.show_camera_feed and not self.camera_window_name:
                            try:
                                self.camera_window_name = "Hand Camera View"
                                cv2.namedWindow(self.camera_window_name, cv2.WINDOW_NORMAL)
                                cv2.resizeWindow(self.camera_window_name, 640, 480)
                            except Exception as e:
                                print(f"Error recreating camera window: {e}")
                                
            # Get controls from camera thread
            if self.camera_thread is not None:
                controls = self.camera_thread.get_controls()
                
                # Get camera visuals
                frame, data_panel, self.fps_camera = self.camera_thread.get_visuals()
                
                # Show camera feed if enabled
                if frame is not None and self.show_camera_feed and self.camera_window_name:
                    try:
                        cv2.imshow(self.camera_window_name, frame)
                        cv2.waitKey(1)
                    except Exception as e:
                        print(f"Error showing camera feed: {e}")
                
                # Update data panel window if needed
                if data_panel is not None and self.show_data_panel:
                    try:
                        cv2.putText(
                            data_panel,
                            f"Game FPS: {self.fps_game:.1f}",
                            (20, 470),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 0),
                            1
                        )
                        cv2.imshow(self.data_panel_cv_window, data_panel)
                        cv2.waitKey(1)
                    except Exception as e:
                        print(f"Error showing data panel: {e}")
                
                # Track last gesture and duration
                current_gesture = controls.get('gesture_name', 'No detection')
                if current_gesture != self.last_gesture:
                    self.gesture_start_time = time.time()
                    self.last_gesture = current_gesture
                else:
                    self.gesture_duration = time.time() - self.gesture_start_time
            else:
                # Default controls if no camera thread
                controls = {
                    'steering': 0,
                    'throttle': 0.5,
                    'braking': False,
                    'boost': False,
                    'gesture_name': 'No camera thread'
                }

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