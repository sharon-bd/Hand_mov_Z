#!/usr/bin/env python
"""
Camera Manager for Hand Gesture Car Control Game

This module handles camera management, camera selection, and processing
of camera frames for hand detection.
"""

import cv2
import pygame
import numpy as np
import os
import time

class CameraManager:
    """Manages camera devices for the game"""
    
    def __init__(self):
        """Initialize the camera manager"""
        # Camera state
        self.current_camera = None
        self.current_camera_index = -1
        self.available_cameras = []
        self.camera_frame = None
        self.camera_enabled = True
        
        # Display settings
        self.display_size = (320, 240)
        self.show_camera = True
        
        # Camera error handling
        self.error_count = 0
        self.max_errors = 10
        self.last_error_time = 0
        
        # Find available cameras
        self.scan_for_cameras()
    
    def scan_for_cameras(self):
        """Scan for available camera devices"""
        print("Scanning for cameras...")
        self.available_cameras = []
        
        # Check first 10 camera indices
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    # Try to read a frame
                    ret, frame = cap.read()
                    if ret:
                        # Camera works, add to list
                        name = f"Camera {i}"
                        
                        # Try to get camera name (works on some systems)
                        try:
                            backend = cap.getBackendName()
                            if backend:
                                name = f"Camera {i} ({backend})"
                        except:
                            pass
                        
                        # Get camera resolution
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
                        self.available_cameras.append({
                            "index": i,
                            "name": name,
                            "resolution": (width, height)
                        })
                        print(f"Found camera {i}: {name}, {width}x{height}")
                
                # Release the camera
                cap.release()
                
            except Exception as e:
                print(f"Error checking camera {i}: {e}")
        
        # Auto-select the first camera if available
        if self.available_cameras and self.current_camera_index < 0:
            self.select_camera(0)
            
        print(f"Found {len(self.available_cameras)} cameras")
        return self.available_cameras
    
    def select_camera(self, index):
        """
        Select and initialize a camera by index
        
        Args:
            index: Index into the available_cameras list
            
        Returns:
            Boolean indicating success
        """
        # Release current camera if any
        self.release_camera()
        
        # Check if index is valid
        if not 0 <= index < len(self.available_cameras):
            print(f"Invalid camera index: {index}")
            return False
        
        try:
            # Get camera device index
            camera_index = self.available_cameras[index]["index"]
            print(f"Selecting camera {camera_index}: {self.available_cameras[index]['name']}")
            
            # Initialize the camera
            self.current_camera = cv2.VideoCapture(camera_index)
            
            # Check if camera opened successfully
            if not self.current_camera.isOpened():
                print(f"Error: Could not open camera {camera_index}")
                self.current_camera = None
                self.current_camera_index = -1
                return False
            
            # Set camera properties
            self.current_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.current_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Update current index
            self.current_camera_index = index
            self.error_count = 0
            
            # Try to read the first frame
            ret, frame = self.current_camera.read()
            if ret:
                print("Camera initialized successfully")
                self.camera_frame = frame
                return True
            else:
                print("Error reading initial frame from camera")
                self.release_camera()
                return False
                
        except Exception as e:
            print(f"Error selecting camera: {e}")
            self.release_camera()
            return False
    
    def release_camera(self):
        """Release the current camera"""
        if self.current_camera is not None:
            self.current_camera.release()
            self.current_camera = None
    
    def get_frame(self):
        """
        Get the latest frame from the camera
        
        Returns:
            The camera frame or None if no camera or error
        """
        if not self.camera_enabled or self.current_camera is None:
            return None
        
        try:
            # Read frame from camera
            ret, frame = self.current_camera.read()
            
            if ret:
                # Reset error count on successful read
                self.error_count = 0
                
                # Store the frame
                self.camera_frame = frame
                
                # Flip frame for mirror effect (more intuitive)
                frame = cv2.flip(frame, 1)
                
                return frame
            else:
                # Increment error count
                self.error_count += 1
                current_time = time.time()
                
                # Limit error messages to avoid spamming
                if current_time - self.last_error_time > 1.0:
                    print(f"Error reading frame from camera. Error count: {self.error_count}")
                    self.last_error_time = current_time
                
                # Check if we should try to recover the camera
                if self.error_count >= self.max_errors:
                    print("Too many camera errors, attempting to recover...")
                    self.recover_camera()
                
                return None
                
        except Exception as e:
            print(f"Error getting camera frame: {e}")
            self.error_count += 1
            
            # Check if we should try to recover the camera
            if self.error_count >= self.max_errors:
                print("Too many camera errors, attempting to recover...")
                self.recover_camera()
                
            return None
    
    def recover_camera(self):
        """Attempt to recover from camera errors"""
        # Release and reinitialize the camera
        self.release_camera()
        
        # Wait a short time before reconnecting
        time.sleep(0.5)
        
        # Try to select the same camera again
        if self.current_camera_index >= 0:
            if self.select_camera(self.current_camera_index):
                print("Camera recovered successfully")
                self.error_count = 0
            else:
                print("Failed to recover camera")
    
    def get_camera_for_display(self):
        """
        Get the latest camera frame formatted for display in Pygame
        
        Returns:
            Pygame surface with the camera image, or None if no image
        """
        if self.camera_frame is None:
            return None
        
        try:
            # Resize the frame for display
            frame = cv2.resize(self.camera_frame, self.display_size)
            
            # Flip for mirror effect (more intuitive)
            frame = cv2.flip(frame, 1)
            
            # Convert from BGR to RGB (Pygame uses RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create Pygame surface from the frame
            frame_surface = pygame.Surface(self.display_size)
            pygame_frame = pygame.image.frombuffer(frame_rgb.tobytes(), self.display_size, "RGB")
            frame_surface.blit(pygame_frame, (0, 0))
            
            return frame_surface
            
        except Exception as e:
            print(f"Error converting camera frame for display: {e}")
            return None
    
    def get_camera_info(self):
        """
        Get information about the current camera
        
        Returns:
            Dictionary with camera information
        """
        if self.current_camera_index < 0 or self.current_camera is None:
            return {
                "available": False,
                "name": "No camera",
                "index": -1,
                "resolution": (0, 0)
            }
        
        return {
            "available": True,
            "name": self.available_cameras[self.current_camera_index]["name"],
            "index": self.current_camera_index,
            "resolution": self.available_cameras[self.current_camera_index]["resolution"]
        }
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        self.camera_enabled = not self.camera_enabled
        
        if not self.camera_enabled:
            self.release_camera()
        elif self.current_camera_index >= 0:
            self.select_camera(self.current_camera_index)
            
        return self.camera_enabled
    
    def cycle_camera(self):
        """Cycle to the next available camera"""
        if not self.available_cameras:
            return False
        
        # Calculate the next index
        next_index = (self.current_camera_index + 1) % len(self.available_cameras)
        
        # Select the next camera
        return self.select_camera(next_index)
    
    def draw_camera_selection(self, screen, font, selected_index=None):
        """
        Draw camera selection interface
        
        Args:
            screen: Pygame surface to draw on
            font: Font to use for text
            selected_index: Currently selected camera index (or None)
        """
        # If selected_index is None, use the current camera
        if selected_index is None:
            selected_index = self.current_camera_index
        
        # Create dark overlay background
        overlay = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))  # Semi-transparent black
        screen.blit(overlay, (0, 0))
        
        # Draw title
        title = font.render("Select Camera", True, (255, 255, 255))
        screen.blit(title, (screen.get_width() // 2 - title.get_width() // 2, 100))
        
        if not self.available_cameras:
            # No cameras available
            no_cam_text = font.render("No cameras detected", True, (255, 0, 0))
            screen.blit(no_cam_text, (screen.get_width() // 2 - no_cam_text.get_width() // 2, 250))
            
            instruction = font.render("Press SPACE to continue without camera", True, (255, 255, 255))
            screen.blit(instruction, (screen.get_width() // 2 - instruction.get_width() // 2, 300))
        else:
            # List available cameras
            y_pos = 180
            for i, camera in enumerate(self.available_cameras):
                color = (0, 255, 0) if i == selected_index else (255, 255, 255)
                camera_text = font.render(camera["name"], True, color)
                
                # Calculate text position
                text_pos = (screen.get_width() // 2 - camera_text.get_width() // 2, y_pos)
                
                # Draw selection indicator
                if i == selected_index:
                    pygame.draw.rect(
                        screen,
                        (0, 100, 0),
                        (text_pos[0] - 10, text_pos[1] - 5, camera_text.get_width() + 20, camera_text.get_height() + 10),
                        0,
                        5
                    )
                
                # Draw text
                screen.blit(camera_text, text_pos)
                y_pos += 40
            
            # Draw camera preview if available
            if selected_index >= 0 and self.camera_frame is not None:
                # Get camera frame for preview
                preview_frame = self.get_camera_for_display()
                
                if preview_frame is not None:
                    # Draw frame in center
                    preview_x = screen.get_width() // 2 - preview_frame.get_width() // 2
                    preview_y = y_pos + 20
                    
                    # Draw border
                    pygame.draw.rect(
                        screen,
                        (0, 255, 0),
                        (preview_x - 2, preview_y - 2, preview_frame.get_width() + 4, preview_frame.get_height() + 4),
                        2,
                        5
                    )
                    
                    # Draw preview
                    screen.blit(preview_frame, (preview_x, preview_y))
                    
                    # Label
                    preview_label = font.render("Camera Preview", True, (255, 255, 255))
                    screen.blit(
                        preview_label,
                        (screen.get_width() // 2 - preview_label.get_width() // 2, preview_y - 30)
                    )
            
            # Draw instructions
            y_pos = screen.get_height() - 100
            instructions = [
                "Use UP/DOWN arrow keys to select a camera",
                "Press SPACE to confirm selection",
                "Press ESC to cancel"
            ]
            
            for instruction in instructions:
                text = font.render(instruction, True, (255, 255, 255))
                screen.blit(text, (screen.get_width() // 2 - text.get_width() // 2, y_pos))
                y_pos += 30
    
    def cleanup(self):
        """Release resources"""
        self.release_camera()