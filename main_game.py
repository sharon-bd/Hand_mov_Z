#!/usr/bin/env python
"""
Game Launcher for Hand Gesture Car Control System
Connects hand gestures to car controls
"""

import os
import sys
import time
import pygame
import cv2
import numpy as np
from hand_detector.connection import HandCarConnectionManager
from game.car import Car

class GameLauncher:
    """Main game launcher class"""
    
    def __init__(self):
        """Initialize the game launcher"""
        pygame.init()
        pygame.display.set_caption("Hand Gesture Car Control")
        
        # Set up the screen
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width + 320, self.screen_height))  # Add width for camera feed
        
        # Set up the clock
        self.clock = pygame.time.Clock()
        self.target_fps = 60
        
        # Create the car
        self.car = Car(self.screen_width // 2, self.screen_height // 2)
        self.car.set_screen_dimensions(self.screen_width, self.screen_height)
        
        # Create the hand-car connection
        self.connection = HandCarConnectionManager()
        
        # Game state
        self.running = True
        self.debug_mode = True
        self.show_help = True
        self.show_camera = True  # Always show camera feed
        
        # Camera display
        self.camera_surface = None
        
        # Fonts
        self.font = pygame.font.SysFont(None, 24)
        self.title_font = pygame.font.SysFont(None, 36)
        
        # Create a separate OpenCV window for the camera feed
        self.show_opencv_window = True
        self.opencv_window_name = "Hand Detector Camera"
        if self.show_opencv_window:
            cv2.namedWindow(self.opencv_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.opencv_window_name, 640, 480)
        
        print("🎮 Game launcher initialized")
        
    def run(self):
        """Run the game loop"""
        # Start the hand-car connection
        if not self.connection.start():
            print("❌ Failed to start hand-car connection")
            return
            
        print("🏁 Starting game loop")
        
        # Main game loop
        last_time = time.time()
        while self.running:
            # Calculate delta time
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_h:
                        self.show_help = not self.show_help
                    elif event.key == pygame.K_d:
                        self.debug_mode = not self.debug_mode
                    elif event.key == pygame.K_c:
                        self.show_camera = not self.show_camera
                    elif event.key == pygame.K_v:
                        self.show_opencv_window = not self.show_opencv_window
                        if self.show_opencv_window:
                            cv2.namedWindow(self.opencv_window_name, cv2.WINDOW_NORMAL)
                            cv2.resizeWindow(self.opencv_window_name, 640, 480)
                        else:
                            cv2.destroyWindow(self.opencv_window_name)
                        
            # Get controls from hand detector
            controls = self.connection.get_controls()
            
            # Get camera feed and data panel
            camera_frame, data_panel, fps = self.connection.get_visuals()
            
            # Convert camera frame to pygame surface if available
            if camera_frame is not None:
                # Display the camera feed in OpenCV window
                if self.show_opencv_window:
                    try:
                        cv2.imshow(self.opencv_window_name, camera_frame)
                        cv2.waitKey(1)
                    except Exception as e:
                        print(f"Error showing OpenCV window: {e}")
                
                # Convert OpenCV frame to Pygame surface for in-game display
                try:
                    # Resize the frame for in-game display
                    small_frame = cv2.resize(camera_frame, (320, 240))
                    # Convert BGR to RGB for Pygame
                    small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    # Create Pygame surface
                    self.camera_surface = pygame.surfarray.make_surface(small_frame_rgb.swapaxes(0, 1))
                except Exception as e:
                    print(f"Error converting camera frame: {e}")
                    self.camera_surface = None
            
            # Update car
            self.car.update(controls, dt)
            
            # Draw everything
            self._draw()
            
            # Cap the frame rate
            self.clock.tick(self.target_fps)
            
        # Clean up
        self.connection.stop()
        cv2.destroyAllWindows()
        pygame.quit()
        
    def _draw(self):
        """Draw the game screen"""
        # Clear the screen
        self.screen.fill((240, 240, 240))
        
        # Draw a simple road/track in the main game area
        pygame.draw.rect(
            self.screen,
            (120, 120, 120),
            (100, 100, self.screen_width - 200, self.screen_height - 200),
            0
        )
        
        pygame.draw.rect(
            self.screen,
            (200, 200, 200),
            (110, 110, self.screen_width - 220, self.screen_height - 220),
            0
        )
        
        # Draw the car
        self.car.draw(self.screen)
        
        # Draw camera feed if available and enabled
        if self.camera_surface is not None and self.show_camera:
            # Draw the camera feed on the right side
            self.screen.blit(self.camera_surface, (self.screen_width, 0))
            
            # Add a border around camera feed
            pygame.draw.rect(
                self.screen,
                (0, 0, 0),
                (self.screen_width, 0, 320, 240),
                2
            )
            
            # Add label
            camera_label = self.font.render("Camera Feed (Press C to toggle)", True, (0, 0, 0))
            self.screen.blit(camera_label, (self.screen_width + 10, 245))
        
        # Draw debug info if enabled
        if self.debug_mode:
            controls = self.connection.get_controls()
            debug_text = [
                f"FPS: {self.clock.get_fps():.1f}",
                f"Gesture: {controls.get('gesture_name', 'Unknown')}",
                f"Steering: {controls.get('steering', 0):.2f}",
                f"Throttle: {controls.get('throttle', 0):.2f}",
                f"Braking: {controls.get('braking', False)}",
                f"Boost: {controls.get('boost', False)}",
                f"Car position: ({self.car.x:.1f}, {self.car.y:.1f})",
                f"Car rotation: {self.car.rotation:.1f}°",
                f"Car speed: {self.car.speed:.2f}"
            ]
            
            for i, text in enumerate(debug_text):
                text_surface = self.font.render(text, True, (0, 0, 0))
                self.screen.blit(text_surface, (10, 10 + i * 25))
                
        # Draw help text if enabled
        if self.show_help:
            help_text = [
                "Controls:",
                "- Move hand left/right: Steer",
                "- Raise/lower hand: Speed up/down",
                "- Make a fist: Brake",
                "- Thumb up: Boost",
                "- Open palm: Stop",
                "",
                "Keys:",
                "- ESC: Quit",
                "- H: Toggle this help",
                "- D: Toggle debug info",
                "- C: Toggle camera feed in game",
                "- V: Toggle separate camera window"
            ]
            
            # Draw help panel on right side, below camera
            if self.show_camera:
                help_y = 280
            else:
                help_y = 10
                
            help_surface = pygame.Surface((320, 300), pygame.SRCALPHA)
            pygame.draw.rect(help_surface, (255, 255, 255, 200), (0, 0, 320, 300), 0)
            pygame.draw.rect(help_surface, (0, 0, 0), (0, 0, 320, 300), 2)
            
            title_surface = self.title_font.render("Help", True, (0, 0, 0))
            help_surface.blit(title_surface, (10, 10))
            
            for i, text in enumerate(help_text):
                text_surface = self.font.render(text, True, (0, 0, 0))
                help_surface.blit(text_surface, (10, 50 + i * 20))
                
            self.screen.blit(help_surface, (self.screen_width, help_y))
            
        # Update the display
        pygame.display.flip()
        
    def cleanup(self):
        """Clean up resources"""
        try:
            self.connection.stop()
        except:
            pass
            
        try:
            cv2.destroyAllWindows()
        except:
            pass
            
        try:
            pygame.quit()
        except:
            pass
            
        print("🧹 Game resources cleaned up")