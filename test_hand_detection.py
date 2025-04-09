#!/usr/bin/env python
"""
Hand Gesture Detection Test Script with External Data Display

This script tests the hand gesture detection functionality in isolation.
It captures video from a webcam, processes frames to detect hand gestures,
and displays the results with all numerical data shown outside the camera feed.
"""

import cv2
import time
import sys
import os
import numpy as np

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our enhanced hand detector
from hand_detector.improved_hand_gesture_detector_fixed import EnhancedHandGestureDetector

def find_available_cameras():
    """Check available camera devices and their indices."""
    available_cameras = []
    
    # Check camera indices 0-9
    for i in range(10):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"Camera index {i} is working")
                    available_cameras.append(i)
                cap.release()
            else:
                print(f"Camera index {i} is not available")
        except Exception as e:
            print(f"Error with camera index {i}: {e}")
    
    return available_cameras

def select_camera(available_cameras):
    """Let the user select a camera from the available ones."""
    if not available_cameras:
        return None
    
    if len(available_cameras) == 1:
        print(f"Only one camera found (index {available_cameras[0]}), using it automatically.")
        return available_cameras[0]
    
    print("\nAvailable cameras:")
    for i, cam_idx in enumerate(available_cameras):
        print(f"{i+1}: Camera index {cam_idx}")
    
    selection = -1
    while selection < 1 or selection > len(available_cameras):
        try:
            selection = int(input(f"\nSelect a camera (1-{len(available_cameras)}): "))
        except ValueError:
            print("Please enter a valid number.")
    
    return available_cameras[selection-1]

def create_display_layout(camera_frame, data_panel):
    """
    Create a combined display layout with camera frame and data panel side by side.
    
    Args:
        camera_frame: The processed camera frame
        data_panel: The numerical data panel
        
    Returns:
        layout: Combined image for display
    """
    # Get dimensions
    cam_h, cam_w = camera_frame.shape[:2]
    panel_h, panel_w = data_panel.shape[:2]
    
    # Create a black canvas with enough space for both
    # Put camera frame on left, data panel on right
    layout_w = cam_w + panel_w
    layout_h = max(cam_h, panel_h)
    layout = np.zeros((layout_h, layout_w, 3), dtype=np.uint8)
    
    # Place camera frame on the left
    layout[0:cam_h, 0:cam_w] = camera_frame
    
    # Place data panel on the right
    layout[0:panel_h, cam_w:cam_w+panel_w] = data_panel
    
    # Add a separation line
    cv2.line(layout, (cam_w, 0), (cam_w, layout_h), (200, 200, 200), 2)
    
    return layout

def main():
    """Main test function with external data display."""
    print("Hand Gesture Detection Test with External Data Display")
    print("====================================================\n")
    
    # Find available cameras
    print("Scanning for available cameras...")
    available_cameras = find_available_cameras()
    
    if not available_cameras:
        print("No cameras found. Please connect a webcam and try again.")
        return
    
    # Let user select a camera
    selected_camera = select_camera(available_cameras)
    if selected_camera is None:
        print("No camera selected. Exiting.")
        return
    
    print(f"\nInitializing camera {selected_camera}...")
    
    # Initialize the webcam
    cap = cv2.VideoCapture(selected_camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print(f"Failed to open camera {selected_camera}. Please try another camera.")
        return
    
    print("Initializing hand gesture detector...")
    
    # Initialize the hand gesture detector
    detector = EnhancedHandGestureDetector()
    print("Detector initialized successfully.")
    
    print("\nPress 'q' or 'ESC' to exit.")
    print("Press 'r' to record detection data to CSV file.")
    
    recording = False
    recording_data = []
    start_time = time.time()
    
    # Main processing loop
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        
        if not ret:
            print("Error reading frame from camera. Trying again...")
            continue
        
        # Flip the frame horizontally (mirror image) for more intuitive interaction
        frame = cv2.flip(frame, 1)
        
        # Process the frame to detect hand gestures
        try:
            # With our updated detector, we get controls, frame, and data_panel
            controls, processed_frame, data_panel = detector.detect_gestures(frame)
            
            # Get a stable command (helps reduce jitter)
            stable_command = detector.get_stable_command()
            
            # Record data if recording is enabled
            if recording and stable_command:
                elapsed_time = time.time() - start_time
                recording_data.append({
                    'time': elapsed_time,
                    'command': stable_command,
                    'steering': controls.get('steering', 0),
                    'throttle': controls.get('throttle', 0),
                    'boost': controls.get('boost', False),
                    'braking': controls.get('braking', False)
                })
            
            # Add recording indicator if recording
            if recording:
                cv2.putText(
                    processed_frame,
                    "RECORDING",
                    (processed_frame.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )
            
            # Create combined display
            display = create_display_layout(processed_frame, data_panel)
            
            # Show the combined display
            cv2.imshow('Hand Gesture Detection Test', display)
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            # Show just the raw frame if there's an error
            cv2.imshow('Hand Gesture Detection Test', frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        
        # 'q' or ESC to quit
        if key == ord('q') or key == 27:
            break
            
        # 'r' to toggle recording
        elif key == ord('r'):
            recording = not recording
            if recording:
                print("Recording started...")
                start_time = time.time()
                recording_data = []
            else:
                print("Recording stopped.")
                # Save recording data to CSV file
                if recording_data:
                    save_recording_data(recording_data)
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Test completed.")

def save_recording_data(data):
    """Save recorded gesture data to a CSV file."""
    import csv
    from datetime import datetime
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gesture_recording_{timestamp}.csv"
    
    try:
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['time', 'command', 'steering', 'throttle', 'boost', 'braking']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in data:
                writer.writerow(row)
            
        print(f"Recording saved to {filename}")
    except Exception as e:
        print(f"Error saving recording: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in main program: {e}")
        import traceback
        traceback.print_exc()