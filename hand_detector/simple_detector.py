import cv2
import mediapipe as mp
import numpy as np

class SimpleHandGestureDetector:
    """Simplified hand gesture detector to fix compatibility issues."""
    
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
        
        # New: Maximum steering angle (degrees from vertical)
        self.max_steering_angle = 90
    
    def detect_gestures(self, frame):
        """
        Detect hand gestures in the given frame and return control signals.
        
        Args:
            frame: CV2 image frame
            
        Returns:
            controls: Dictionary with control values
            processed_frame: Frame with visualization of detected hands
            data_panel: Additional image panel with numerical data
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
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
            
            # Draw hand landmarks and extract control information
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks on the frame
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Extract control values from hand landmarks
                    controls = self._extract_controls_from_landmarks(hand_landmarks, frame, controls)
            else:
                # Reset stability counter when no hand detected
                self.command_stability_count = 0
            
            # Create separate data panel
            data_panel = self._create_data_panel(controls)
            
            # Add control mappings
            controls['speed'] = controls['throttle']
            controls['direction'] = controls['steering']
            
            return controls, frame, data_panel
        
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
        print("RUNNING UPDATED CODE VERSION!")
        h, w, c = frame.shape
        landmark_points = []
        
        # Convert landmarks to points
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
        
        # Get knuckle positions
        thumb_mcp = landmark_points[2]
        index_mcp = landmark_points[5]
        middle_mcp = landmark_points[9]
        ring_mcp = landmark_points[13]
        pinky_mcp = landmark_points[17]
        
        # ==================== STEERING DETECTION ====================
        # Calculate thumb angle
        dx_thumb = float(thumb_tip[0] - wrist[0])
        dy_thumb = float(thumb_tip[1] - wrist[1])
        
        # Calculate angle in degrees (0° is up, 90° is right, 270° is left)
        thumb_angle = float(np.degrees(np.arctan2(dx_thumb, -dy_thumb)))
        # Convert to 0-360 range
        if thumb_angle < 0:
            thumb_angle += 360
        
        # Define invalid angle range for thumb pointing down
        INVALID_ANGLE_MIN = 80
        INVALID_ANGLE_MAX = 280
        
        # Check if thumb is pointing in a valid steering direction
        is_valid_steering_angle = not (INVALID_ANGLE_MIN < thumb_angle < INVALID_ANGLE_MAX)
        
        # Additional check: vertical alignment of thumb
        VERTICAL_THRESHOLD = 30  # Margin around vertical (degrees)
        is_too_vertical = (thumb_angle < VERTICAL_THRESHOLD or thumb_angle > 360 - VERTICAL_THRESHOLD or
                          abs(thumb_angle - 180) < VERTICAL_THRESHOLD)
        
        # Debug prints for thumb angle calculation
        print(f"Thumb angle: {thumb_angle:.1f}")
        print(f"Is valid steering angle: {is_valid_steering_angle}")
        print(f"Is too vertical: {is_too_vertical}")
        print(f"Using for steering: {is_valid_steering_angle and not is_too_vertical}")
        
        # Add strong visual emphasis for invalid thumb positions
        if not (is_valid_steering_angle and not is_too_vertical):
            # Draw a large red circle to indicate invalid thumb position
            cv2.circle(frame, thumb_tip, 30, (0, 0, 255), 2)
            # Add large warning text
            cv2.putText(frame, "INVALID THUMB ANGLE", 
                        (frame.shape[1]//2 - 150, frame.shape[0]//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        if is_valid_steering_angle and not is_too_vertical:
            # Convert angle to normalized_angle in range -90 to 90
            if 270 <= thumb_angle <= 360:
                normalized_angle = thumb_angle - 360  # Convert 270-360° to -90-0°
            else:
                normalized_angle = thumb_angle  # 0-90° remains the same
            
            # Map normalized_angle in range -90° to 90° to steering coefficient in range -1.0 to 1.0
            raw_steering = normalized_angle / 90.0
        else:
            # Thumb is pointing down or too vertical - not considered for steering
            raw_steering = 0.0
        
        # Add debug visualization to show valid/invalid regions
        if self.debug_mode:
            thumb_line_length = 100
            thumb_angle_rad = np.radians(thumb_angle)
            thumb_line_end = (
                int(wrist[0] + thumb_line_length * np.sin(thumb_angle_rad)),
                int(wrist[1] - thumb_line_length * np.cos(thumb_angle_rad))
            )
            
            # Color thumb line green if valid angle, red if not
            line_color = (0, 255, 0) if is_valid_steering_angle and not is_too_vertical else (0, 0, 255)
            cv2.line(frame, wrist, thumb_line_end, line_color, 2)
            
            cv2.putText(frame, f"Thumb Angle: {thumb_angle:.1f}°", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)
            cv2.putText(frame, f"Valid Steering: {is_valid_steering_angle and not is_too_vertical}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)
            cv2.putText(frame, f"Steering: {raw_steering:.2f}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Apply smoothing
        steering = float(self.prev_steering * self.steering_smoothing + raw_steering * (1 - self.steering_smoothing))
        steering = float(max(-1.0, min(1.0, steering)))
        self.prev_steering = steering
        controls['steering'] = steering
        controls['thumb_angle'] = thumb_angle  # Add thumb angle to controls
        
        # ==================== THROTTLE DETECTION ====================
        normalized_y = float(1.0 - (wrist[1] / h))
        raw_throttle = normalized_y
        
        # Apply non-linear mapping
        raw_throttle = float(raw_throttle ** 1.5)
        
        # Apply smoothing
        throttle = float(self.prev_throttle * self.throttle_smoothing + raw_throttle * (1 - self.throttle_smoothing))
        throttle = float(max(0.0, min(1.0, throttle)))
        self.prev_throttle = throttle
        controls['throttle'] = throttle
        
        # ==================== GESTURE DETECTION ====================
        index_curled = index_tip[1] > index_mcp[1]
        middle_curled = middle_tip[1] > middle_mcp[1]
        ring_curled = ring_tip[1] > ring_mcp[1]
        pinky_curled = pinky_tip[1] > pinky_mcp[1]
        thumb_curled = thumb_tip[0] > thumb_mcp[0] if wrist[0] > thumb_mcp[0] else thumb_tip[0] < thumb_mcp[0]
        
        v_sign = not index_curled and not middle_curled and ring_curled and pinky_curled
        fist_detected = index_curled and middle_curled and ring_curled and pinky_curled and thumb_curled
        fingers_extended = (
            not index_curled and
            not middle_curled and
            not ring_curled and
            not pinky_curled
        )
        
        stop_sign_gesture = fingers_extended and not thumb_curled
        
        if self.debug_mode:
            cv2.putText(frame, f"Fingers Extended: {fingers_extended}", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"V Sign Detected: {v_sign}", (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Fist Detected: {fist_detected}", (10, 180), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
        boost_gesture = v_sign
        brake_gesture = fist_detected
        
        if stop_sign_gesture:
            controls['gesture_name'] = 'Stop'
            controls['braking'] = True
            controls['throttle'] = 0.0
            controls['boost'] = False
            self._update_command_stability("STOP")
        elif boost_gesture:
            controls['gesture_name'] = 'Boost (V sign)'
            controls['boost'] = True
            controls['braking'] = False
            controls['throttle'] = 1.0
            self._update_command_stability("FORWARD_BOOST")
        elif brake_gesture:
            controls['gesture_name'] = 'Brake'
            controls['braking'] = True
            controls['boost'] = False
            controls['throttle'] = 0.0
            self._update_command_stability("STOP")
        else:
            controls['braking'] = False
            controls['boost'] = False
            
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
                
        controls['thumb_angle'] = thumb_angle
        
        cv2.putText(
            frame,
            f"Gesture: {controls['gesture_name']}",
            (w//2 - 100, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        cv2.putText(
            frame,
            f"Thumb Angle: {thumb_angle:.1f}°",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
                
        return controls
    
    def _update_command_stability(self, command):
        """Track command stability to avoid jitter."""
        if command == self.last_command:
            self.command_stability_count += 1
        else:
            self.last_command = command
            self.command_stability_count = 1
    
    def get_stable_command(self):
        """Get current command only if stable enough."""
        if self.command_stability_count >= self.stability_threshold:
            return self.last_command
        return None
    
    def _create_data_panel(self, controls):
        """Create a panel with control data."""
        panel_width = 400
        panel_height = 300
        panel = np.ones((panel_height, panel_width, 3), dtype=np.uint8) * 255
        
        cv2.putText(panel, "Hand Gesture Controls", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.line(panel, (20, 40), (panel_width - 20, 40), (0, 0, 0), 1)
        
        if 'thumb_angle' in controls:
            thumb_angle = controls['thumb_angle']
            cv2.putText(panel, f"THUMB ANGLE: {thumb_angle:.1f}°", (20, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        
        y_pos = 100
        cv2.putText(panel, "GESTURE:", (20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(panel, f"{controls['gesture_name']}", (150, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_pos += 30
        
        cv2.putText(panel, f"Stability: {self.command_stability_count}/{self.stability_threshold}", 
                    (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_pos += 30
        
        cv2.putText(panel, f"Steering: {controls['steering']:.2f}", (20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        bar_x = 200
        bar_width = 150
        bar_height = 15
        cv2.rectangle(panel, (bar_x, y_pos - 12), (bar_x + bar_width, y_pos + 3), (220, 220, 220), -1)
        cv2.rectangle(panel, (bar_x, y_pos - 12), (bar_x + bar_width, y_pos + 3), (0, 0, 0), 1)
        cv2.line(panel, (bar_x + bar_width//2, y_pos - 15), 
                 (bar_x + bar_width//2, y_pos + 6), (0, 0, 0), 1)
        steer_pos = int(bar_x + bar_width//2 + controls['steering'] * bar_width/2)
        cv2.circle(panel, (steer_pos, y_pos - 5), 6, (0, 0, 255), -1)
        y_pos += 30
        
        cv2.putText(panel, f"Throttle: {controls['throttle']:.2f}", (20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.rectangle(panel, (bar_x, y_pos - 12), (bar_x + bar_width, y_pos + 3), (220, 220, 220), -1)
        cv2.rectangle(panel, (bar_x, y_pos - 12), (bar_x + bar_width, y_pos + 3), (0, 0, 0), 1)
        fill_width = int(bar_width * controls['throttle'])
        cv2.rectangle(panel, (bar_x, y_pos - 12), (bar_x + fill_width, y_pos + 3), (0, 255, 0), -1)
        y_pos += 30
        
        brake_color = (0, 0, 255) if controls['braking'] else (150, 150, 150)
        boost_color = (255, 165, 0) if controls['boost'] else (150, 150, 150)
        cv2.putText(panel, "Braking:", (20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.circle(panel, (120, y_pos - 5), 8, brake_color, -1)
        cv2.putText(panel, "Boost:", (200, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.circle(panel, (260, y_pos - 5), 8, boost_color, -1)
        y_pos += 40
        
        cv2.putText(panel, "CONTROLS:", (20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_pos += 25
        cv2.putText(panel, "- Move hand left/right to steer", (30, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_pos += 20
        cv2.putText(panel, "- Move hand up/down for speed", (30, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_pos += 20
        cv2.putText(panel, "- Make a fist to brake", (30, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_pos += 20
        cv2.putText(panel, "- V sign for boost", (30, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_pos += 20
        cv2.putText(panel, "- Open palm for emergency stop", (30, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return panel
    
    def _create_error_panel(self):
        """Create an error panel when detection fails."""
        panel = np.ones((300, 400, 3), dtype=np.uint8) * 255
        cv2.putText(panel, "ERROR IN HAND DETECTION", (30, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(panel, "Please make sure your hand", (30, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(panel, "is clearly visible in the camera", (30, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        return panel