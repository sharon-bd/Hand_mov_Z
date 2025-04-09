#!/usr/bin/env python
"""
Hand Gesture Visualization Tool

This script provides a simple visualization tool for hand gestures,
allowing you to see how the detection system interprets your hand movements
without needing to run the full game.

It's helpful for practice and debugging hand detection issues.
"""

import cv2
import time
import numpy as np
import sys
import os

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our hand detector
from hand_detector.improved_hand_gesture_detector_fixed import EnhancedHandGestureDetector

def find_available_cameras():
    """Check which camera indices are available."""
    available_cameras = []
    
    # Check camera indices 0-9
    for i in range(10):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"Camera index {i} is available")
                    available_cameras.append(i)
                cap.release()
            else:
                print(f"Camera index {i} is not available")
        except Exception as e:
            print(f"Error checking camera {i}: {e}")
    
    return available_cameras

def select_camera(available_cameras):
    """Let the user select a camera from available ones."""
    if not available_cameras:
        return None
        
    if len(available_cameras) == 1:
        return available_cameras[0]
    
    print("\nAvailable cameras:")
    for i, cam_idx in enumerate(available_cameras):
        print(f"{i+1}. Camera index {cam_idx}")
    
    try:
        selection = int(input(f"Select a camera (1-{len(available_cameras)}): "))
        if 1 <= selection <= len(available_cameras):
            return available_cameras[selection-1]
    except ValueError:
        pass
    
    print("Invalid selection. Using first available camera.")
    return available_cameras[0]

def create_visual_feedback(controls, width=400, height=400):
    """Create a visual representation of the current controls."""
    # Create a white background
    visualization = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Extract control values
    steering = controls.get('steering', 0)
    throttle = controls.get('throttle', 0)
    braking = controls.get('braking', False)
    boost = controls.get('boost', False)
    gesture_name = controls.get('gesture_name', 'Unknown')
    
    # Draw title
    cv2.putText(
        visualization,
        "Control Visualization",
        (width//2 - 120, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2
    )
    
    # Draw car representation
    car_width = 60
    car_height = 100
    car_x = width // 2
    car_y = height // 2
    
    # Draw the car with steering visualized by rotation
    # Create car points
    car_points = np.array([
        [-car_width//2, -car_height//2],  # Top-left
        [car_width//2, -car_height//2],   # Top-right
        [car_width//2, car_height//2],    # Bottom-right
        [-car_width//2, car_height//2]    # Bottom-left
    ], np.int32)
    
    # Rotate points based on steering
    angle_rad = steering * np.pi/4  # Convert steering to radians (max ±45°)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    # Apply rotation
    rotated_points = []
    for point in car_points:
        rotated = np.dot(rotation_matrix, point)
        rotated_points.append([int(rotated[0] + car_x), int(rotated[1] + car_y)])
    
    # Car color based on boost/brake status
    car_color = (0, 0, 200)  # Default blue
    if boost:
        car_color = (0, 165, 255)  # Orange for boost
    elif braking:
        car_color = (0, 0, 255)    # Red for braking
    
    # Draw car body
    cv2.fillPoly(visualization, [np.array(rotated_points)], car_color)
    cv2.polylines(visualization, [np.array(rotated_points)], True, (0, 0, 0), 2)
    
    # Draw windshield
    windshield_points = np.array([
        [-car_width//3, -car_height//2 + 15],
        [car_width//3, -car_height//2 + 15],
        [car_width//3, -car_height//5],
        [-car_width//3, -car_height//5]
    ], np.int32)
    
    # Rotate windshield points
    rotated_windshield = []
    for point in windshield_points:
        rotated = np.dot(rotation_matrix, point)
        rotated_windshield.append([int(rotated[0] + car_x), int(rotated[1] + car_y)])
    
    cv2.fillPoly(visualization, [np.array(rotated_windshield)], (150, 230, 255))
    
    # Draw steering indicator
    cv2.putText(
        visualization,
        f"Steering: {steering:.2f}",
        (20, height - 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1
    )
    
    # Steering bar
    bar_width = 200
    bar_height = 20
    bar_x = width//2 - bar_width//2
    bar_y = height - 100
    
    # Draw bar background
    cv2.rectangle(
        visualization,
        (bar_x, bar_y),
        (bar_x + bar_width, bar_y + bar_height),
        (220, 220, 220),
        -1
    )
    cv2.rectangle(
        visualization,
        (bar_x, bar_y),
        (bar_x + bar_width, bar_y + bar_height),
        (0, 0, 0),
        1
    )
    
    # Draw center line
    center_x = bar_x + bar_width//2
    cv2.line(
        visualization,
        (center_x, bar_y),
        (center_x, bar_y + bar_height),
        (0, 0, 0),
        1
    )
    
    # Draw steering indicator
    indicator_pos = int(center_x + steering * bar_width/2)
    cv2.rectangle(
        visualization,
        (indicator_pos - 5, bar_y),
        (indicator_pos + 5, bar_y + bar_height),
        (0, 0, 255),
        -1
    )
    
    # Draw throttle indicator
    cv2.putText(
        visualization,
        f"Throttle: {throttle:.2f}",
        (20, height - 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1
    )
    
    # Throttle bar
    t_bar_width = 20
    t_bar_height = 150
    t_bar_x = width - 50
    t_bar_y = height//2 - t_bar_height//2
    
    # Draw bar background
    cv2.rectangle(
        visualization,
        (t_bar_x, t_bar_y),
        (t_bar_x + t_bar_width, t_bar_y + t_bar_height),
        (220, 220, 220),
        -1
    )
    cv2.rectangle(
        visualization,
        (t_bar_x, t_bar_y),
        (t_bar_x + t_bar_width, t_bar_y + t_bar_height),
        (0, 0, 0),
        1
    )
    
    # Draw throttle fill
    fill_height = int(throttle * t_bar_height)
    cv2.rectangle(
        visualization,
        (t_bar_x, t_bar_y + t_bar_height - fill_height),
        (t_bar_x + t_bar_width, t_bar_y + t_bar_height),
        (0, 255, 0),
        -1
    )
    
    # Draw current gesture
    cv2.putText(
        visualization,
        f"Gesture: {gesture_name}",
        (width//2 - 80, height - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2
    )
    
    # Draw boost/brake indicators
    cv2.putText(
        visualization,
        "BOOST" if boost else "",
        (width - 100, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 165, 255) if boost else (200, 200, 200),
        2
    )
    
    cv2.putText(
        visualization,
        "BRAKE" if braking else "",
        (width - 100, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255) if braking else (200, 200, 200),
        2
    )
    
    return visualization

def main():
    """Run the hand gesture visualization tool."""
    print("Hand Gesture Visualization Tool")
    print("===============================")
    
    # Find available cameras
    print("\nScanning for available cameras...")
    available_cameras = find_available_cameras()
    
    if not available_cameras:
        print("No cameras found. Please connect a webcam and try again.")
        return
    
    # Select camera
    camera_idx = select_camera(available_cameras)
    if camera_idx is None:
        print("No camera selected. Exiting.")
        return
    
    print(f"\nInitializing camera {camera_idx}...")
    
    # Initialize webcam
    cap = cv2.VideoCapture(camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print(f"Failed to open camera {camera_idx}. Please try another camera.")
        return
    
    # Initialize hand detector
    print("Initializing hand gesture detector...")
    detector = EnhancedHandGestureDetector()
    
    # Create windows
    cv2.namedWindow("Hand Detection", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Control Visualization", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Data Panel", cv2.WINDOW_NORMAL)
    
    print("\nPress 'q' or ESC to exit.")
    
    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame from camera.")
                break
            
            # Flip horizontally for more intuitive interaction
            frame = cv2.flip(frame, 1)
            
            # Process frame for hand gestures
            controls, processed_frame, data_panel = detector.detect_gestures(frame)
            
            # Create visual feedback
            visualization = create_visual_feedback(controls)
            
            # Show all windows
            cv2.imshow("Hand Detection", processed_frame)
            cv2.imshow("Control Visualization", visualization)
            cv2.imshow("Data Panel", data_panel)
            
            # Check for exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("Visualization tool closed.")

if __name__ == "__main__":
    main()