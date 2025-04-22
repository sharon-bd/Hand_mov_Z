import cv2
import mediapipe as mp
import numpy as np

class SimpleHandGestureDetector:
    """Simple hand gesture detector with fixed thumb detection."""
    
    def __init__(self):
        """Initialize the hand gesture detector with MediaPipe."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Control state with smoothing
        self.prev_steering = 0
        self.prev_throttle = 0
        self.steering_smoothing = 0.5
        self.throttle_smoothing = 0.4
        
        # State tracking
        self.last_command = None
        self.command_stability_count = 0
        self.stability_threshold = 3
        
        # Debug mode
        self.debug_mode = True
        
        # FPS calculation
        self.prev_frame_time = 0
        self.curr_frame_time = 0
        self.fps = 0.0
    
    def detect_gestures(self, frame):
        """
        Detect hand gestures in the given frame and return control signals.
        
        Args:
            frame: CV2 image frame
            
        Returns:
            controls: Dictionary with control values
            processed_frame: Frame with visualization
            data_panel: Additional data visualization
        """
        try:
            # Update FPS calculation
            self.curr_frame_time = cv2.getTickCount()
            if self.prev_frame_time > 0:
                self.fps = cv2.getTickFrequency() / (self.curr_frame_time - self.prev_frame_time)
            self.prev_frame_time = self.curr_frame_time
            
            # First create a mirrored copy of the input frame for better user experience
            mirrored_frame = cv2.flip(frame.copy(), 1)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe
            results = self.hands.process(rgb_frame)
            
            # Default controls
            controls = {
                'steering': 0.0,
                'throttle': 0.0,
                'braking': False,
                'boost': False,
                'gesture_name': 'No hand detected',
                'speed': 0.0,
                'direction': 0.0
            }
            
            # Draw hand landmarks and extract controls
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks on the frame
                    self.mp_draw.draw_landmarks(
                        mirrored_frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Extract controls from landmarks
                    controls = self._extract_controls_from_landmarks(hand_landmarks, mirrored_frame, controls)
            else:
                # Reset stability counter when no hand detected
                self.command_stability_count = 0
            
            # Add speed and direction for compatibility
            controls['speed'] = controls['throttle']
            controls['direction'] = controls['steering']
            
            # Create data panel
            data_panel = self._create_data_panel(controls)
            
            # Add FPS display
            cv2.putText(
                mirrored_frame,
                f"FPS: {self.fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
            
            return controls, mirrored_frame, data_panel
            
        except Exception as e:
            print(f"Error in gesture detection: {e}")
            import traceback
            traceback.print_exc()
            return {
                'steering': 0.0,
                'throttle': 0.0,
                'braking': False,
                'boost': False,
                'gesture_name': 'Error',
                'speed': 0.0,
                'direction': 0.0
            }, frame, self._create_error_panel()
    
    def _extract_controls_from_landmarks(self, landmarks, frame, controls):
        """Extract control values from hand landmarks."""
        h, w, c = frame.shape
        landmark_points = []
        
        # Convert landmarks to accessible format
        for lm in landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            landmark_points.append((x, y))
        
        # Get key points
        wrist = landmark_points[0]
        thumb_tip = landmark_points[4]
        index_tip = landmark_points[8]
        middle_tip = landmark_points[12]
        ring_tip = landmark_points[16]
        pinky_tip = landmark_points[20]
        
        # Get MCP (knuckle) positions
        thumb_mcp = landmark_points[2]
        index_mcp = landmark_points[5]
        middle_mcp = landmark_points[9]
        ring_mcp = landmark_points[13]
        pinky_mcp = landmark_points[17]
        
        # ==================== STEERING DETECTION ====================
        # Calculate hand tilt for steering
        dx = landmark_points[17][0] - landmark_points[5][0]  # Pinky MCP - Index MCP x-distance
        dy = landmark_points[17][1] - landmark_points[5][1]  # Pinky MCP - Index MCP y-distance
        
        # Calculate angle and map to steering value (-1 to 1)
        hand_angle = np.degrees(np.arctan2(dy, dx))
        
        # Map angle to steering (-135° to -45° maps to -1 to 1)
        if -135 <= hand_angle <= -45:
            raw_steering = (hand_angle + 90) / 45
        else:
            raw_steering = -1 if hand_angle < -135 else 1
        
        # Apply smoothing for more stable steering
        steering = self.prev_steering * self.steering_smoothing + raw_steering * (1 - self.steering_smoothing)
        steering = max(-1.0, min(1.0, steering))  # Clamp to valid range
        self.prev_steering = steering
        controls['steering'] = steering
        
        # ==================== THROTTLE DETECTION ====================
        # Detect throttle based on hand height (y-position)
        normalized_y = 1.0 - (wrist[1] / h)  # Invert so higher hand = lower value
        raw_throttle = normalized_y
        
        # Apply non-linear mapping for better control
        raw_throttle = raw_throttle ** 1.5
        
        # Apply smoothing
        throttle = self.prev_throttle * self.throttle_smoothing + raw_throttle * (1 - self.throttle_smoothing)
        throttle = max(0.0, min(1.0, throttle))  # Clamp to valid range
        self.prev_throttle = throttle
        controls['throttle'] = throttle
        
        # ==================== GESTURE DETECTION ====================
        # Check if fingers are curled
        index_curled = index_tip[1] > index_mcp[1]
        middle_curled = middle_tip[1] > middle_mcp[1]
        ring_curled = ring_tip[1] > ring_mcp[1]
        pinky_curled = pinky_tip[1] > pinky_mcp[1]
        
        # Thumb position check
        thumb_extended = self._is_thumb_extended(thumb_tip, wrist, thumb_mcp)
        
        # Detect gestures
        fist_gesture = index_curled and middle_curled and ring_curled and pinky_curled and not thumb_extended
        open_palm = not index_curled and not middle_curled and not ring_curled and not pinky_curled
        thumb_up_gesture = thumb_extended and index_curled and middle_curled and ring_curled and pinky_curled
        
        # Set control values based on gestures
        if open_palm:
            controls['gesture_name'] = 'Stop (Open Palm)'
            controls['braking'] = True
            controls['throttle'] = 0.0
            controls['boost'] = False
            self._update_command_stability("STOP")
        elif fist_gesture:
            controls['gesture_name'] = 'Brake (Fist)'
            controls['braking'] = True
            controls['throttle'] = 0.0
            controls['boost'] = False
            self._update_command_stability("BRAKE")
        elif thumb_up_gesture:
            controls['gesture_name'] = 'Boost (Thumb Up)'
            controls['boost'] = True
            controls['braking'] = False
            controls['throttle'] = 1.0
            self._update_command_stability("BOOST")
        else:
            # Regular driving with steering and throttle
            if abs(steering) > 0.3:
                if steering < -0.3:
                    controls['gesture_name'] = 'Turning Left'
                    self._update_command_stability("LEFT")
                else:
                    controls['gesture_name'] = 'Turning Right'
                    self._update_command_stability("RIGHT")
            else:
                controls['gesture_name'] = 'Forward'
                self._update_command_stability("FORWARD")
        
        # Add gesture name to frame
        cv2.putText(
            frame,
            f"Gesture: {controls['gesture_name']}",
            (frame.shape[1]//2 - 100, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        return controls
    
    def _is_thumb_extended(self, thumb_tip, wrist, thumb_mcp):
        """Check if thumb is extended upward."""
        # Basic check if thumb is pointing up
        basic_check = thumb_tip[1] < wrist[1] - (wrist[1] - thumb_mcp[1]) * 0.2
        
        # Additional horizontal distance check
        horizontal_distance = abs(thumb_tip[0] - wrist[0])
        
        # Thumb is up if it's both above and sufficiently far from wrist
        return basic_check and horizontal_distance > 20
    
    def _update_command_stability(self, command):
        """Track command stability to avoid jitter."""
        if command == self.last_command:
            self.command_stability_count += 1
        else:
            self.last_command = command
            self.command_stability_count = 1
    
    def _create_data_panel(self, controls):
        """Create a data panel with control information."""
        panel_width = 500
        panel_height = 400
        panel = np.ones((panel_height, panel_width, 3), dtype=np.uint8) * 255
        
        # Add title
        cv2.putText(
            panel,
            "Hand Gesture Controls",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2
        )
        cv2.line(panel, (20, 40), (panel_width - 20, 40), (0, 0, 0), 1)
        
        # Current gesture
        cv2.putText(
            panel,
            f"Gesture: {controls['gesture_name']}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )
        
        # Control values
        y_pos = 110
        cv2.putText(
            panel,
            f"Steering: {controls['steering']:.2f}",
            (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1
        )
        y_pos += 30
        
        cv2.putText(
            panel,
            f"Throttle: {controls['throttle']:.2f}",
            (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1
        )
        y_pos += 30
        
        # Status indicators
        brake_status = "Active" if controls['braking'] else "Inactive"
        boost_status = "Active" if controls['boost'] else "Inactive"
        
        cv2.putText(
            panel,
            f"Brake: {brake_status}",
            (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255) if controls['braking'] else (0, 0, 0),
            1
        )
        y_pos += 30
        
        cv2.putText(
            panel,
            f"Boost: {boost_status}",
            (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 165, 0) if controls['boost'] else (0, 0, 0),
            1
        )
        y_pos += 40
        
        # Instructions
        cv2.putText(
            panel,
            "Control Instructions:",
            (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            1
        )
        y_pos += 30
        
        instructions = [
            "- Tilt hand left/right to steer",
            "- Raise/lower hand for throttle",
            "- Make a fist to brake",
            "- Show thumb up for boost",
            "- Open palm for emergency stop"
        ]
        
        for instruction in instructions:
            cv2.putText(
                panel,
                instruction,
                (30, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
            y_pos += 25
        
        return panel
    
    def _create_error_panel(self):
        """Create an error panel when detection fails."""
        panel = np.ones((300, 400, 3), dtype=np.uint8) * 255
        cv2.putText(
            panel,
            "ERROR IN HAND DETECTION",
            (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
        cv2.putText(
            panel,
            "Please make sure your hand is",
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1
        )
        cv2.putText(
            panel,
            "clearly visible in the camera.",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1
        )
        return panel
        
    def release(self):
        """Release resources."""
        pass  # No need to release resources for this version
