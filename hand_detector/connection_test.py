"""
Testing communication between gesture recognition and car control
This file allows isolated testing of communication between modules
"""

import cv2
import time
import numpy as np
import sys
import os
import argparse

# Add the main library to the search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import hand gesture detection module and any relevant car module
from hand_detector.gestures import HandGestureDetector

def parse_args():
    """Parse command line arguments for window size"""
    parser = argparse.ArgumentParser(description='Hand gesture detection test')
    parser.add_argument('--width', type=int, default=800, help='Initial window width')
    parser.add_argument('--height', type=int, default=600, help='Initial window height')
    return parser.parse_args()

def test_hand_gesture_controls():
    """
    Test gesture recognition and conversion to control values
    """
    args = parse_args()
    
    print("Hand Gesture Recognition System Test")
    print("====================================")
    
    # Create hand gesture detector
    detector = HandGestureDetector()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open the camera")
        return
    
    # Update camera parameters
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Create resizable window
    cv2.namedWindow("Hand Gesture Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hand Gesture Test", args.width, args.height)
    
    print("Running test... Move your hand in front of the camera")
    print("Controls:")
    print("- Press 'q' to exit")
    print("- Press '+' to increase window size")
    print("- Press '-' to decrease window size")
    print("- Press 'r' to reset window size")
    
    # Default window size for reset
    default_width = args.width
    default_height = args.height
    current_width = default_width
    current_height = default_height
    
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame from camera")
            break
        
        # Detect gestures and convert to control values
        start_time = time.time()
        controls, processed_frame = detector.detect_gestures(frame)
        processing_time = time.time() - start_time
        
        # Calculate FPS once
        fps = 1.0 / processing_time if processing_time > 0 else 0
        
        # IMPORTANT: At this point, processed_frame already contains the visualization from
        # the gesture detector. We'll add only a few additional UI elements here to avoid duplication
        
        # Add FPS counter at the top-right corner (small overlay)
        cv2.rectangle(processed_frame, (processed_frame.shape[1]-150, 0), 
                     (processed_frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(processed_frame, f"FPS: {fps:.1f}", 
                    (processed_frame.shape[1]-140, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add warning only if both controls are at maximum (potential issue)
        if controls['steering'] == 1.0 and controls['throttle'] == 1.0:
            warning_overlay = processed_frame.copy()
            # Place warning above the bottom panel but below other UI elements
            warning_y = processed_frame.shape[0] - 150
            cv2.rectangle(warning_overlay, (50, warning_y-30), (500, warning_y+10), (0, 0, 0), -1)
            cv2.addWeighted(warning_overlay, 0.7, processed_frame, 0.3, 0, processed_frame)
            
            cv2.putText(processed_frame, "WARNING: Both controls at maximum!", 
                      (60, warning_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        # Show processed image
        cv2.imshow("Hand Gesture Test", processed_frame)
        
        # Print DEBUG information to terminal only
        print(f"DEBUG - Steering: {controls['steering']:.2f}, Throttle: {controls['throttle']:.2f}, "
              f"Gesture: {controls['gesture_name']}")
        
        # Handle keyboard commands
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+'):
            # Increase window size by 10%
            current_width = int(current_width * 1.1)
            current_height = int(current_height * 1.1)
            cv2.resizeWindow("Hand Gesture Test", current_width, current_height)
            print(f"Window resized to: {current_width}x{current_height}")
        elif key == ord('-'):
            # Decrease window size by 10%
            current_width = max(320, int(current_width * 0.9))
            current_height = max(240, int(current_height * 0.9))
            cv2.resizeWindow("Hand Gesture Test", current_width, current_height)
            print(f"Window resized to: {current_width}x{current_height}")
        elif key == ord('r'):
            # Reset to default size
            current_width = default_width
            current_height = default_height
            cv2.resizeWindow("Hand Gesture Test", current_width, current_height)
            print(f"Window reset to: {current_width}x{current_height}")
                   
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Test completed")

if __name__ == "__main__":
    test_hand_gesture_controls()
