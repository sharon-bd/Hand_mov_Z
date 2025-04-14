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
        self.prev_steering = 0.0
        self.steering_smoothing = 0.5
        # Add throttle smoothing parameters
        self.prev_throttle = 0.0
        self.throttle_smoothing = 0.5
        
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
                
                # Extract thumb tip position
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                thumb_tip_x = thumb_tip.x * width
                thumb_tip_y = thumb_tip.y * height
                
                # ==================== STEERING DETECTION ====================
                # Calculate thumb angle
                dx_thumb = float(thumb_tip_x - wrist_x)
                dy_thumb = float(thumb_tip_y - wrist_y)

                # Calculate angle in degrees (0° is up, positive clockwise)
                thumb_angle = float(np.degrees(np.arctan2(dx_thumb, -dy_thumb)))
                # Convert to 0-360 range
                if thumb_angle < 0:
                    thumb_angle += 360

                # Print debug info about thumb angle
                if self.debug_mode:
                    print(f"Thumb angle: {thumb_angle:.1f}")

                # Define valid angle ranges for steering
                LEFT_RANGE_START = 225  # 270° - 45°
                LEFT_RANGE_END = 315    # 270° + 45°
                RIGHT_RANGE_START = 45  # 90° - 45°
                RIGHT_RANGE_END = 135   # 90° + 45°
                UP_RANGE_START = 315    # 360° - 45°
                UP_RANGE_END = 45       # 0° + 45°

                # Check if the angle is in any of the valid steering ranges
                is_left = LEFT_RANGE_START <= thumb_angle <= LEFT_RANGE_END
                is_right = RIGHT_RANGE_START <= thumb_angle <= RIGHT_RANGE_END
                is_up = thumb_angle >= UP_RANGE_START or thumb_angle <= UP_RANGE_END

                # Combine the checks to get the final valid flag
                is_valid_steering_angle = is_left or is_right or is_up

                # Calculate steering value based on the angle range
                if is_valid_steering_angle:
                    if is_left:
                        # Map 225°-315° to -1.0 (full left) to -0.0
                        normalized_angle = (thumb_angle - LEFT_RANGE_START) / (LEFT_RANGE_END - LEFT_RANGE_START)
                        raw_steering = -1.0 + normalized_angle  # Maps from -1.0 to 0.0
                    elif is_right:
                        # Map 45°-135° to 0.0 to 1.0 (full right)
                        normalized_angle = (thumb_angle - RIGHT_RANGE_START) / (RIGHT_RANGE_END - RIGHT_RANGE_START)
                        raw_steering = normalized_angle  # Maps from 0.0 to 1.0
                    elif is_up:
                        # Special case for up (wraps around 360)
                        if thumb_angle >= UP_RANGE_START:
                            # Map 315°-360° to -0.33 to 0.0
                            normalized_angle = (thumb_angle - UP_RANGE_START) / (360 - UP_RANGE_START)
                            raw_steering = -0.33 + normalized_angle * 0.33  # Maps to small negative values around 0
                        else:  # thumb_angle <= UP_RANGE_END
                            # Map 0°-45° to 0.0 to 0.33
                            normalized_angle = thumb_angle / UP_RANGE_END
                            raw_steering = normalized_angle * 0.33  # Maps to small positive values around 0
                    else:
                        # Shouldn't happen, but default to 0 for safety
                        raw_steering = 0.0
                else:
                    # Not in valid range, no steering
                    raw_steering = 0.0

                # Add debug visualization to show valid/invalid regions
                if self.debug_mode:
                    thumb_line_length = 100
                    thumb_angle_rad = np.radians(thumb_angle)
                    thumb_line_end = (
                        int(wrist_x + thumb_line_length * np.sin(thumb_angle_rad)),
                        int(wrist_y - thumb_line_length * np.cos(thumb_angle_rad))
                    )
                    
                    # Color thumb line based on range
                    if is_left:
                        line_color = (0, 0, 255)  # Red for left
                    elif is_right:
                        line_color = (0, 255, 0)  # Green for right
                    elif is_up:
                        line_color = (255, 255, 0)  # Yellow for up
                    else:
                        line_color = (128, 128, 128)  # Gray for invalid
                    
                    cv2.line(processed_frame, (int(wrist_x), int(wrist_y)), thumb_line_end, line_color, 2)
                    
                    cv2.putText(processed_frame, f"Thumb Angle: {thumb_angle:.1f}°", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)
                    cv2.putText(processed_frame, f"Valid Steering: {is_valid_steering_angle}", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)
                    cv2.putText(processed_frame, f"Steering: {raw_steering:.2f}", (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Apply smoothing
                steering = float(self.prev_steering * self.steering_smoothing + raw_steering * (1 - self.steering_smoothing))
                steering = float(max(-1.0, min(1.0, steering)))  # Clamp to valid range
                self.prev_steering = steering
                self.controls['steering'] = steering
                self.controls['thumb_angle'] = thumb_angle  # Add thumb angle to controls
                
                # ==================== THROTTLE DETECTION ====================
                # Normalize wrist position on Y-axis (screen height)
                normalized_y = float(1.0 - (wrist.y))  # Invert value so higher = higher hand
                raw_throttle = normalized_y

                # Non-linear mapping for better control (exponential for finer control at low speeds)
                raw_throttle = float(raw_throttle ** 1.5)

                # Smooth movement and clamp to valid range
                throttle = float(self.prev_throttle * self.throttle_smoothing + raw_throttle * (1 - self.throttle_smoothing))
                throttle = float(max(0.0, min(1.0, throttle)))  # Clamp to range 0 to 1
                self.prev_throttle = throttle
                self.controls['throttle'] = throttle

                # Add debug visualization if required
                if self.debug_mode:
                    throttle_line_y = int(height * (1 - normalized_y))
                    cv2.line(processed_frame, (0, throttle_line_y), (width, throttle_line_y), (0, 255, 0), 2)
                    cv2.putText(processed_frame, f"Throttle: {throttle:.2f}", (10, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
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
