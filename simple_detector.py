#!/usr/bin/env python
"""
Simple Hand Gesture Detector

A simplified version of the hand gesture detector that uses OpenCV and MediaPipe
for detecting hand positions and gestures.
"""

import cv2
import mediapipe as mp
import numpy as np
import time

class SimpleHandGestureDetector:
    """A simplified hand gesture detector class."""
    
    def __init__(self, camera_index=0):
        """Initialize the hand detector.
        
        Args:
            camera_index (int): The index of the camera to use
        """
        self.camera_index = camera_index
        self.cap = None
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Open the camera
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera at index {self.camera_index}")
        
        # Initialize last frame time for FPS calculation
        self.last_frame_time = time.time()
        self.fps = 0
    
    def detect(self):
        """Detect hand position and gestures.
        
        Returns:
            dict: Hand data including position and gesture, or None if no hand found
        """
        if self.cap is None or not self.cap.isOpened():
            return None
        
        # Read a frame from the camera
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Calculate FPS
        current_time = time.time()
        time_diff = current_time - self.last_frame_time
        if time_diff > 0:
            self.fps = 1.0 / time_diff
        self.last_frame_time = current_time
        
        # Flip the image horizontally for a more intuitive mirror view
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands
        results = self.hands.process(rgb_frame)
        
        # Initialize hand data
        hand_data = None
        
        # Check if a hand was detected
        if results.multi_hand_landmarks:
            # We only process the first hand for simplicity
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Get image dimensions
            h, w, _ = frame.shape
            
            # Get hand positions (normalized coordinates)
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            
            # Determine gesture based on finger positions
            gesture = self._determine_gesture(hand_landmarks)
            
            # Create hand data dictionary with normalized positions
            hand_data = {
                'normalized_position_x': wrist.x,  # Value between 0 and 1
                'normalized_position_y': wrist.y,  # Value between 0 and 1
                'gesture': gesture
            }
            
            # Draw hand landmarks on the frame for visualization
            self._draw_landmarks(frame, hand_landmarks)
            
            # Add gesture text
            cv2.putText(
                frame,
                f"Gesture: {gesture}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
        
        # Show the FPS
        cv2.putText(
            frame,
            f"FPS: {self.fps:.1f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Display the frame
        cv2.imshow('Hand Gesture Detector', frame)
        cv2.waitKey(1)  # Update window with minimal delay
        
        return hand_data
    
    def _determine_gesture(self, hand_landmarks):
        """Determine the gesture from hand landmarks.
        
        Args:
            hand_landmarks: The detected hand landmarks
            
        Returns:
            str: The detected gesture ('open_palm', 'fist', 'pointing', 'thumb_up', 'steering')
        """
        # Get the relevant landmarks
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        
        # Get the middle knuckle positions for comparison
        thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
        index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP]
        pinky_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]
        
        # Check if fingers are extended by comparing tip to MCP (knuckle)
        # For y-coordinate in image, lower value is higher position
        thumb_extended = thumb_tip.y < thumb_ip.y
        index_extended = index_tip.y < index_mcp.y
        middle_extended = middle_tip.y < middle_mcp.y
        ring_extended = ring_tip.y < ring_mcp.y
        pinky_extended = pinky_tip.y < pinky_mcp.y
        
        # Calculate hand tilt (for steering)
        hand_tilt = self._calculate_hand_tilt(hand_landmarks)
        
        # Logic to determine the gesture based on SRS document requirements
        if index_extended and middle_extended and ring_extended and pinky_extended:
            return 'open_palm'  # Emergency stop as per SRS
        elif not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return 'fist'  # Brake as per SRS
        elif index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return 'pointing'  # Special command as per SRS
        elif thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return 'thumb_up'  # Boost as per SRS
        else:
            # Default to steering/throttle control based on hand position
            return 'steering'
    
    def _calculate_hand_tilt(self, hand_landmarks):
        """Calculate the tilt of the hand for steering purposes.
        
        Args:
            hand_landmarks: The detected hand landmarks
            
        Returns:
            float: Tilt angle in degrees
        """
        # Use index finger MCP and pinky MCP to determine hand tilt
        index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        pinky_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]
        
        # Calculate the angle
        dx = pinky_mcp.x - index_mcp.x
        dy = pinky_mcp.y - index_mcp.y
        
        # Convert to degrees (0 is horizontal)
        angle = np.degrees(np.arctan2(dy, dx))
        
        return angle
    
    def _draw_landmarks(self, frame, hand_landmarks):
        """Draw hand landmarks on the frame.
        
        Args:
            frame: The video frame to draw on
            hand_landmarks: The detected hand landmarks
        """
        h, w, _ = frame.shape
        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )
    
    def release(self):
        """Release resources."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
