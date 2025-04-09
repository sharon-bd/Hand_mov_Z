#!/usr/bin/env python
"""
Camera Utility Functions

This module provides utilities for camera management including:
- Finding available cameras
- Testing camera functionality
- Selecting between multiple cameras
"""

import cv2
import time
import pygame
import numpy as np

def get_available_cameras(max_cameras=10):
    """
    Get a list of available camera devices on the system.
    
    Args:
        max_cameras (int): Maximum number of cameras to check
        
    Returns:
        list: List of dictionaries containing camera information
    """
    available_cameras = []
    
    # Try each camera index
    for camera_index in range(max_cameras):
        cap = cv2.VideoCapture(camera_index)
        
        # Check if camera opened successfully
        if cap.isOpened():
            # Try to read a frame
            ret, frame = cap.read()
            
            if ret:
                # Camera works, add to list with details
                camera_info = {
                    "index": camera_index,
                    "name": f"Camera {camera_index}",
                    "resolution": (
                        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    )
                }
                
                # Try to get more camera details
                try:
                    backend = cap.getBackendName()
                    if backend:
                        camera_info["name"] = f"Camera {camera_index} ({backend})"
                except:
                    pass
                
                available_cameras.append(camera_info)
        
        # Release the camera
        cap.release()
    
    return available_cameras

def test_camera(camera_index=0, display_time=3):
    """
    Test a specific camera and display a preview.
    
    Args:
        camera_index (int): Index of the camera to test
        display_time (int): Time in seconds to show the preview
        
    Returns:
        bool: True if camera works, False otherwise
    """
    # Try to open the camera
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Failed to open camera {camera_index}")
        return False
    
    # Create a window for preview
    window_name = f"Testing Camera {camera_index}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    start_time = time.time()
    camera_working = False
    
    while time.time() - start_time < display_time:
        ret, frame = cap.read()
        
        if ret:
            camera_working = True
            # Show the frame
            cv2.imshow(window_name, frame)
            # Display countdown
            remaining = int(display_time - (time.time() - start_time))
            print(f"Camera {camera_index} test - closing in {remaining} seconds", end="\r")
            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print(f"Failed to read frame from camera {camera_index}")
            break
    
    # Clean up
    cap.release()
    cv2.destroyWindow(window_name)
    print("")  # New line after countdown
    
    return camera_working

def camera_selection_gui():
    """
    Launch a standalone GUI for camera selection.
    
    Returns:
        int: Selected camera index or -1 if cancelled
    """
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Camera Selection")
    
    # Fonts
    font_large = pygame.font.Font(None, 48)
    font_medium = pygame.font.Font(None, 36)
    font_small = pygame.font.Font(None, 24)
    
    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    
    # Get available cameras
    cameras = get_available_cameras()
    current_selection = 0 if cameras else -1
    
    # Camera preview
    preview_cap = None
    preview_surface = None
    
    # Main loop
    running = True
    selected_camera = -1
    
    while running:
        # Fill screen
        screen.fill((30, 30, 50))
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_RETURN and current_selection >= 0:
                    # Camera selected
                    selected_camera = cameras[current_selection]["index"]
                    running = False
                elif event.key == pygame.K_UP and current_selection > 0:
                    current_selection -= 1
                elif event.key == pygame.K_DOWN and current_selection < len(cameras) - 1:
                    current_selection += 1
        
        # Draw title
        title_text = font_large.render("Camera Selection", True, WHITE)
        screen.blit(title_text, (400 - title_text.get_width()//2, 50))
        
        if not cameras:
            # No cameras found
            no_cam_text = font_medium.render("No cameras detected", True, RED)
            screen.blit(no_cam_text, (400 - no_cam_text.get_width()//2, 250))
        else:
            # List cameras
            y_pos = 150
            for i, camera in enumerate(cameras):
                color = GREEN if i == current_selection else WHITE
                text = font_medium.render(camera["name"], True, color)
                screen.blit(text, (400 - text.get_width()//2, y_pos))
                y_pos += 40
            
            # Update camera preview
            if preview_cap is None or i != current_selection:
                # Open new camera for preview
                if preview_cap is not None:
                    preview_cap.release()
                
                camera_index = cameras[current_selection]["index"]
                preview_cap = cv2.VideoCapture(camera_index)
            
            # Get frame from selected camera
            if preview_cap.isOpened():
                ret, frame = preview_cap.read()
                if ret:
                    # Resize and convert frame for display
                    frame = cv2.flip(frame, 1)
                    frame = cv2.resize(frame, (320, 240))
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert to pygame surface
                    preview_surface = pygame.Surface((320, 240))
                    preview_image = pygame.image.frombuffer(frame_rgb.tobytes(), (320, 240), "RGB")
                    preview_surface.blit(preview_image, (0, 0))
            
            # Draw preview
            if preview_surface is not None:
                screen.blit(preview_surface, (240, 300))
                pygame.draw.rect(screen, GREEN, (238, 298, 324, 244), 2)
                
                preview_label = font_small.render("Camera Preview", True, WHITE)
                screen.blit(preview_label, (400 - preview_label.get_width()//2, 280))
        
        # Draw instructions
        instructions = font_small.render("UP/DOWN: Select camera | ENTER: Confirm | ESC: Cancel", True, WHITE)
        screen.blit(instructions, (400 - instructions.get_width()//2, 550))
        
        # Update display
        pygame.display.flip()
    
    # Clean up
    if preview_cap is not None:
        preview_cap.release()
    pygame.quit()
    
    return selected_camera

if __name__ == "__main__":
    # If run directly, show available cameras and allow testing
    cameras = get_available_cameras()
    
    if not cameras:
        print("No cameras detected on your system.")
    else:
        print(f"Found {len(cameras)} camera(s):")
        for camera in cameras:
            print(f"  {camera['index']}: {camera['name']}, Resolution: {camera['resolution']}")
        
        # Allow camera testing
        print("\nWould you like to test a camera? (y/n)")
        if input().lower().startswith('y'):
            print("Enter camera index to test:")
            try:
                index = int(input())
                test_camera(index)
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        # Launch camera selection GUI
        print("\nLaunching camera selection GUI...")
        selected = camera_selection_gui()
        if selected >= 0:
            print(f"Selected camera index: {selected}")
        else:
            print("No camera selected.")
