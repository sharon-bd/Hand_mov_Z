#!/usr/bin/env python
"""
Hand Gesture Detection Test Script

This script tests the hand gesture detection functionality in isolation.
It captures video from a webcam, processes frames to detect hand gestures,
and displays the results in real-time.
"""

import cv2
import time
import sys
import os

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our hand detector
from hand_detector.improved_hand_gesture_detector import EnhancedHandGestureDetector

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

def main():
    """Main test function."""
    print("Hand Gesture Detection Test")
    print("==========================\n")
    
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
            controls, processed_frame = detector.detect_gestures(frame)
            
            # Get a stable command (helps reduce jitter)
            stable_command = detector.get_stable_command()
            
            # Display the command on the frame
            if stable_command:
                cv2.putText(
                    processed_frame,
                    f"Command: {stable_command}",
                    (20, processed_frame.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                # Record data if recording is enabled
                if recording:
                    elapsed_time = time.time() - start_time
                    recording_data.append({
                        'time': elapsed_time,
                        'command': stable_command,
                        'steering': controls.get('steering', 0),
                        'throttle': controls.get('throttle', 0),
                        'boost': controls.get('boost', False),
                        'braking': controls.get('braking', False)
                    })
            
            # Add debug information to frame
            cv2.putText(
                processed_frame,
                "Press 'q' or ESC to exit",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
            
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
            
            # Show the frame with hand tracking visualization
            cv2.imshow('Hand Gesture Detection Test', processed_frame)
            
        except Exception as e:
            print(f"Error processing frame: {e}")
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