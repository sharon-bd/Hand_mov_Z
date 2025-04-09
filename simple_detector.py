#!/usr/bin/env python
"""
Simple Hand Gesture Detector

This module provides a simplified hand gesture detection implementation 
for the car control game. It detects basic hand positions and gestures
to control car movement.

Key features:
- Hand position detection (left/right, up/down)
- Gesture recognition (open palm, fist, etc.)
- Visual feedback for debugging
"""

import cv2
import numpy as np
import mediapipe as mp
import time

class SimpleHandGestureDetector:
    """A simplified hand gesture detector for controlling a car in a game."""
    
    def __init__(self):
        """Initialize the hand gesture detector with MediaPipe."""
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize hand position tracking
        self.prev_hand_x = None
        self.prev_hand_y = None
        self.smoothing_factor = 0.5
        
        # Control state
        self.controls = {
            'steering': 0.0,     # -1.0 to 1.0 (left to right)
            'throttle': 0.0,     # 0.0 to 1.0
            'braking': False,    # Boolean
            'boost': False       # Boolean
        }
        
        # Debugging and feedback
        self.debug_mode = True
        
    def detect_gestures(self, frame):
        """
        Process a video frame to detect hand gestures and return control values.
        
        Args:
            frame (numpy.ndarray): The input video frame
        
        Returns:
            tuple: (controls, processed_frame, data_panel)
                - controls: Dictionary of control values (steering, throttle, etc.)
                - processed_frame: Frame with hand landmarks drawn
                - data_panel: Visualization of control values
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands
        results = self.hands.process(image_rgb)
        
        # Create a copy of the frame for drawing
        processed_frame = frame.copy()
        height, width, _ = processed_frame.shape
        
        # Reset controls to default values
        self.controls = {
            'steering': 0.0,
            'throttle': 0.0,
            'braking': False,
            'boost': False
        }
        
        # If hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    processed_frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS)
                
                # Extract wrist position (base of hand)
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                wrist_x = wrist.x * width
                wrist_y = wrist.y * height
                
                # Apply smoothing if we have previous positions
                if self.prev_hand_x is not None and self.prev_hand_y is not None:
                    wrist_x = self.prev_hand_x + (wrist_x - self.prev_hand_x) * self.smoothing_factor
                    wrist_y = self.prev_hand_y + (wrist_y - self.prev_hand_y) * self.smoothing_factor
                
                # Store current position for next frame
                self.prev_hand_x = wrist_x
                self.prev_hand_y = wrist_y
                
                # Calculate steering (x position)
                # Map hand x position from 0.1-0.9 to -1.0-1.0
                norm_x = max(0.0, min(1.0, (wrist.x - 0.1) / 0.8))
                steering = (norm_x * 2.0) - 1.0
                self.controls['steering'] = steering
                
                # Calculate throttle (y position)
                # Map hand y position from 0.2-0.8 to 0.0-1.0 (inverted, lower hand is more throttle)
                norm_y = max(0.0, min(1.0, (wrist.y - 0.2) / 0.6))
                throttle = 1.0 - norm_y
                self.controls['throttle'] = throttle
                
                # Detect gestures for boost and brake
                # Calculate average distance between fingertips and palm
                palm = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                fingertips = [
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP],
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP],
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP],
                    hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
                ]
                
                # Calculate average distance from palm to fingertips
                total_dist = 0
                for fingertip in fingertips:
                    dist = ((fingertip.x - palm.x)**2 + (fingertip.y - palm.y)**2)**0.5
                    total_dist += dist
                avg_dist = total_dist / 5
                
                # Detect fist gesture (small distance = fist)
                fist_threshold = 0.08
                if avg_dist < fist_threshold:
                    # Hand is in a fist
                    if wrist.y < 0.4:  # Hand is high - boost
                        self.controls['boost'] = True
                        cv2.putText(processed_frame, "BOOST", (50, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    else:  # Hand is low/middle - brake
                        self.controls['braking'] = True
                        cv2.putText(processed_frame, "BRAKE", (50, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Detect open palm (emergency stop)
                palm_threshold = 0.15
                if avg_dist > palm_threshold:
                    # Emergency stop - override all controls
                    self.controls['steering'] = 0.0
                    self.controls['throttle'] = 0.0
                    self.controls['braking'] = True
                    self.controls['boost'] = False
                    cv2.putText(processed_frame, "STOP", (50, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Create data panel
        data_panel = self._create_data_panel(width, height)
        
        return self.controls, processed_frame, data_panel
    
    def _create_data_panel(self, width, height):
        """Create a visualization panel for control values."""
        # Create a blank image for the data panel
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Fill background
        panel.fill(50)  # Dark gray background
        
        # Draw steering indicator
        cv2.putText(panel, "Steering", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        # Draw steering bar
        bar_width = int(width - 40)
        bar_height = 20
        cv2.rectangle(panel, (20, 40), (20 + bar_width, 40 + bar_height), (100, 100, 100), -1)
        # Draw steering position
        steering_pos = int(20 + bar_width * ((self.controls['steering'] + 1) / 2))
        cv2.rectangle(panel, (steering_pos - 5, 35), (steering_pos + 5, 65), (0, 255, 0), -1)
        
        # Draw throttle indicator
        cv2.putText(panel, "Throttle", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        # Draw throttle bar
        cv2.rectangle(panel, (20, 100), (20 + bar_width, 100 + bar_height), (100, 100, 100), -1)
        # Draw throttle position
        throttle_width = int(bar_width * self.controls['throttle'])
        throttle_color = (0, 255, 255) if self.controls['boost'] else (0, 255, 0)
        cv2.rectangle(panel, (20, 100), (20 + throttle_width, 100 + bar_height), throttle_color, -1)
        
        # Draw brake indicator
        brake_color = (0, 0, 255) if self.controls['braking'] else (100, 100, 100)
        cv2.putText(panel, "Brake", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, brake_color, 2)
        
        # Draw boost indicator
        boost_color = (0, 255, 255) if self.controls['boost'] else (100, 100, 100)
        cv2.putText(panel, "Boost", (width - 100, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, boost_color, 2)
        
        # Draw gesture guide
        cv2.putText(panel, "Hand left/right: Steer", (20, 190), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(panel, "Hand up/down: Throttle", (20, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(panel, "Fist: Brake", (20, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(panel, "Fist up high: Boost", (20, 280), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(panel, "Open palm: Emergency stop", (20, 310), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return panel

# Simple test function
def main():
    """Test the SimpleHandGestureDetector with webcam feed."""
    cap = cv2.VideoCapture(0)
    detector = SimpleHandGestureDetector()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame with detector
        controls, processed_frame, data_panel = detector.detect_gestures(frame)
        
        # Display controls
        print(f"\rSteering: {controls['steering']:.2f} | Throttle: {controls['throttle']:.2f} | "
              f"Brake: {controls['braking']} | Boost: {controls['boost']}      ", end="")
        
        # Show output
        cv2.imshow("Hand Tracking", processed_frame)
        cv2.imshow("Control Data", data_panel)
        
        # Exit on ESC key
        if cv2.waitKey(1) == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
