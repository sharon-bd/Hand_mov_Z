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
        thumb_angle = float(np.degrees(np.arctan2(dx_thumb, -dy_thumb)))
        
        # Map angle to steering
        if -90 <= thumb_angle <= 90:
            raw_steering = float(thumb_angle / 90.0)
        else:
            raw_steering = -1.0 if thumb_angle < -90 else 1.0
            
        # Apply smoothing
        steering = float(self.prev_steering * self.steering_smoothing + raw_steering * (1 - self.steering_smoothing))
        steering = float(max(-1.0, min(1.0, steering)))
        self.prev_steering = steering
        controls['steering'] = steering
        
        # Draw indicator for thumb angle in debug mode
        if self.debug_mode:
            thumb_line_length = 100
            thumb_angle_rad = np.radians(thumb_angle)
            thumb_line_end = (
                int(wrist[0] + thumb_line_length * np.sin(thumb_angle_rad)),
                int(wrist[1] - thumb_line_length * np.cos(thumb_angle_rad))
            )
            cv2.line(frame, wrist, thumb_line_end, (255, 0, 255), 2)
            cv2.putText(frame, f"Thumb Angle: {thumb_angle:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(frame, f"Steering: {steering:.2f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
        # ==================== THROTTLE DETECTION ====================
        # Throttle based on hand height (y-position)
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
        # Check if fingers are curled
        index_curled = index_tip[1] > index_mcp[1]
        middle_curled = middle_tip[1] > middle_mcp[1]
        ring_curled = ring_tip[1] > ring_mcp[1]
        pinky_curled = pinky_tip[1] > pinky_mcp[1]
        thumb_curled = thumb_tip[0] > thumb_mcp[0] if wrist[0] > thumb_mcp[0] else thumb_tip[0] < thumb_mcp[0]
        
        # Fist detection
        fist_detected = index_curled and middle_curled and ring_curled and pinky_curled and thumb_curled
        
        # Check if fingers are extended
        fingers_extended = (
            not index_curled and
            not middle_curled and
            not ring_curled and
            not pinky_curled
        )
        
        # Simple stop gesture detection
        stop_sign_gesture = fingers_extended and not thumb_curled
        
        # Add debug visualization
        if self.debug_mode:
            cv2.putText(frame, f"Fingers Extended: {fingers_extended}", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Fist Detected: {fist_detected}", (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
        # Detect boost and brake gestures
        boost_gesture = fist_detected and wrist[1] < h/2
        brake_gesture = fist_detected and wrist[1] >= h/2
        
        # Set controls based on detected gestures
        if stop_sign_gesture:
            controls['gesture_name'] = 'Stop'
            controls['braking'] = True
            controls['throttle'] = 0.0
            controls['boost'] = False
            self._update_command_stability("STOP")
        elif boost_gesture:
            controls['gesture_name'] = 'Boost'
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
            # Regular driving with steering and throttle
            controls['braking'] = False
            controls['boost'] = False
            
            if abs(steering) > 0.3:  # Significant steering
                if steering < -0.3:
                    controls['gesture_name'] = 'Turning Left'
                    self._update_command_stability("LEFT")
                else:
                    controls['gesture_name'] = 'Turning Right'
                    self._update_command_stability("RIGHT")
            else:
                controls['gesture_name'] = 'Forward'
                self._update_command_stability("FORWARD")
        
        # Draw current gesture
        cv2.putText(
            frame,
            f"Gesture: {controls['gesture_name']}",
            (w//2 - 100, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
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
        
        # Add title
        cv2.putText(panel, "Hand Gesture Controls", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.line(panel, (20, 40), (panel_width - 20, 40), (0, 0, 0), 1)
        
        # Add current gesture info
        y_pos = 80
        cv2.putText(panel, "GESTURE:", (20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(panel, f"{controls['gesture_name']}", (150, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_pos += 30
        
        # Add stability info
        cv2.putText(panel, f"Stability: {self.command_stability_count}/{self.stability_threshold}", 
                    (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_pos += 30
        
        # Add steering info
        cv2.putText(panel, f"Steering: {controls['steering']:.2f}", (20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Draw steering bar
        bar_x = 200
        bar_width = 150
        bar_height = 15
        cv2.rectangle(panel, (bar_x, y_pos - 12), (bar_x + bar_width, y_pos + 3), (220, 220, 220), -1)
        cv2.rectangle(panel, (bar_x, y_pos - 12), (bar_x + bar_width, y_pos + 3), (0, 0, 0), 1)
        
        # Center line
        cv2.line(panel, (bar_x + bar_width//2, y_pos - 15), 
                (bar_x + bar_width//2, y_pos + 6), (0, 0, 0), 1)
        
        # Position indicator
        steer_pos = int(bar_x + bar_width//2 + controls['steering'] * bar_width/2)
        cv2.circle(panel, (steer_pos, y_pos - 5), 6, (0, 0, 255), -1)
        y_pos += 30
        
        # Add throttle info
        cv2.putText(panel, f"Throttle: {controls['throttle']:.2f}", (20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Draw throttle bar
        cv2.rectangle(panel, (bar_x, y_pos - 12), (bar_x + bar_width, y_pos + 3), (220, 220, 220), -1)
        cv2.rectangle(panel, (bar_x, y_pos - 12), (bar_x + bar_width, y_pos + 3), (0, 0, 0), 1)
        
        # Fill bar based on throttle value
        fill_width = int(bar_width * controls['throttle'])
        cv2.rectangle(panel, (bar_x, y_pos - 12), (bar_x + fill_width, y_pos + 3), (0, 255, 0), -1)
        y_pos += 30
        
        # Add brake and boost status
        brake_color = (0, 0, 255) if controls['braking'] else (150, 150, 150)
        boost_color = (255, 165, 0) if controls['boost'] else (150, 150, 150)
        
        cv2.putText(panel, "Braking:", (20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.circle(panel, (120, y_pos - 5), 8, brake_color, -1)
        
        cv2.putText(panel, "Boost:", (200, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.circle(panel, (260, y_pos - 5), 8, boost_color, -1)
        y_pos += 40
        
        # Add control help
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
        cv2.putText(panel, "- Fist up high for boost", (30, y_pos), 
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