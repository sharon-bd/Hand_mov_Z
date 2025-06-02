import cv2
import mediapipe as mp
import numpy as np
import time  # Add time for debugging timestamps

class HandGestureDetector:
    """Class to detect hand gestures and convert them to car control signals 
    according to the SRS specifications."""
    
    def __init__(self):
        """Initialize the hand gesture detector with MediaPipe."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Track only one hand for simplicity
            min_detection_confidence=0.5,  # Reduced for better detection rate
            min_tracking_confidence=0.5    # Better tracking
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Control state with improved smoothing
        self.prev_steering = 0
        self.prev_throttle = 0
        self.steering_smoothing = 0.5  # Balanced value for stable but responsive steering
        self.throttle_smoothing = 0.4  # Slightly faster response for throttle
        
        # State tracking for gesture stability
        self.gesture_history = []
        self.history_size = 5
        self.last_command = None
        self.command_stability_count = 0
        self.stability_threshold = 3  # Require this many consistent readings
        
        # Debug and display options
        self.debug_mode = True
        self.last_log_time = time.time()
        self.log_interval = 2.0  # Log every 2 seconds
        
        # Throttle calibration
        self.min_hand_height = 1.0  # Will be updated during detection
        self.max_hand_height = 0.0  # Will be updated during detection
        self.height_calibration_alpha = 0.1  # Smoothing factor for height range updates
        
        # Fallback controls in case of detection issues
        self.fallback_controls = {
            'steering': 0.0,
            'throttle': 0.0,
            'braking': False,
            'boost': False,
            'gesture_name': 'Fallback controls',
            'speed': 0.0,
            'direction': 0.0
        }
        
        print("🖐️ Hand gesture detector initialized - v2.0 with improved reliability")
        
    def ensure_valid_frame(self, frame):
        """Ensure we have a valid frame to process"""
        if frame is None:
            print("⚠️ Warning: Received None frame, using blank frame instead")
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Check if frame is grayscale and convert to BGR if needed
        if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
            print("⚠️ Warning: Received grayscale frame, converting to BGR")
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
        # Check if frame is too small
        if frame.shape[0] < 10 or frame.shape[1] < 10:
            print(f"⚠️ Warning: Frame too small {frame.shape}, using blank frame")
            return np.zeros((480, 640, 3), dtype=np.uint8)
            
        return frame
        
    def detect_gestures(self, frame):
        """
        Detect hand gestures in the given frame and return control signals.
        
        Args:
            frame: CV2 image frame
            
        Returns:
            controls: Dictionary with control values
            processed_frame: Frame with visualization
        """
        try:
            # Ensure we have a valid frame
            frame = self.ensure_valid_frame(frame)
            
            # Periodic debugging
            current_time = time.time()
            if current_time - self.last_log_time > self.log_interval:
                self.last_log_time = current_time
                print(f"🖐️ Hand detector processing frame: shape={frame.shape}")
            
            # Flip the frame horizontally for more natural interaction
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe
            results = self.hands.process(rgb_frame)
            
            # Default controls
            controls = {
                'steering': 0.0,     # -1.0 (full left) to 1.0 (full right)
                'throttle': 0.0,     # 0.0 to 1.0
                'braking': False,
                'boost': False,
                'gesture_name': 'No hand detected'
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
                    
                    # Log detected gestures at intervals
                    if current_time - self.last_log_time > self.log_interval:
                        print(f"👋 Detected gesture: {controls['gesture_name']}, " +
                              f"Steering: {controls['steering']:.2f}, " +
                              f"Throttle: {controls['throttle']:.2f}")
            else:
                # Reset stability counter when no hand detected
                self.command_stability_count = 0
                
            # Add visualization of current controls to the frame
            self._add_control_visualization(frame, controls)
            
            # Add speed and direction mappings for compatibility with the Car class
            controls['speed'] = controls['throttle']
            controls['direction'] = controls['steering']
            
            return controls, frame
            
        except Exception as e:
            print(f"Error in gesture detection: {e}")
            import traceback
            traceback.print_exc()
            # Return the fallback controls to ensure the car still gets input
            return self.fallback_controls, frame
    
    def _extract_controls_from_landmarks(self, landmarks, frame, controls):
        """
        Extract control values from hand landmarks according to SRS specs:
        - Steering: hand tilt for left/right
        - Throttle: hand height for speed
        - Brake: fist gesture
        - Boost: thumb up, other fingers curled
        - Stop: open palm
        """
        # Convert landmarks to more accessible format
        h, w, c = frame.shape
        landmark_points = []
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
        
        # Get MCP (knuckle) positions for detecting finger curling
        thumb_mcp = landmark_points[2]
        index_mcp = landmark_points[5]
        middle_mcp = landmark_points[9]
        ring_mcp = landmark_points[13]
        pinky_mcp = landmark_points[17]
        
        # ==================== STEERING DETECTION ====================
        dx = thumb_tip[0] - wrist[0]
        dy = thumb_tip[1] - wrist[1]
        
        thumb_angle = np.degrees(np.arctan2(-dy, dx))
        if thumb_angle < 0:
            thumb_angle += 360
        
        if 80 <= thumb_angle <= 100:
            raw_steering = 0.0
        elif thumb_angle < 80:
            if thumb_angle < 45:
                raw_steering = 1.0
            else:
                raw_steering = 0.3 + 0.7 * (80 - thumb_angle) / 35
        elif thumb_angle > 100:
            if thumb_angle > 135:
                raw_steering = -1.0
            else:
                raw_steering = -0.3 - 0.7 * (thumb_angle - 100) / 35

        steering = self.prev_steering * self.steering_smoothing + raw_steering * (1 - self.steering_smoothing)
        steering = max(-1.0, min(1.0, steering))
        self.prev_steering = steering
        controls['steering'] = steering
        
        # ==================== THROTTLE DETECTION ====================
        # Get normalized hand height (0 at top of frame, 1 at bottom)
        normalized_y = wrist[1] / h
        
        # Update height calibration range with smoothing
        if normalized_y < self.min_hand_height:
            self.min_hand_height = (1 - self.height_calibration_alpha) * self.min_hand_height + self.height_calibration_alpha * normalized_y
        if normalized_y > self.max_hand_height:
            self.max_hand_height = (1 - self.height_calibration_alpha) * self.max_hand_height + self.height_calibration_alpha * normalized_y
        
        # Calculate throttle based on calibrated range
        height_range = self.max_hand_height - self.min_hand_height
        if height_range > 0:
            raw_throttle = 1.0 - (normalized_y - self.min_hand_height) / height_range
        else:
            raw_throttle = 1.0 - normalized_y  # Fallback if range not established
            
        # Apply non-linear mapping for better control
        raw_throttle = raw_throttle ** 1.5  # More precise control at lower speeds
        
        # Apply smoothing
        throttle = self.prev_throttle * self.throttle_smoothing + raw_throttle * (1 - self.throttle_smoothing)
        throttle = max(0.0, min(1.0, throttle))
        self.prev_throttle = throttle
        controls['throttle'] = throttle
        
        # ==================== GESTURE DETECTION ====================
        index_curled = index_tip[1] > index_mcp[1]
        middle_curled = middle_tip[1] > middle_mcp[1]
        ring_curled = ring_tip[1] > ring_mcp[1]
        pinky_curled = pinky_tip[1] > pinky_mcp[1]
        
        # בדיקה משופרת של האגודל
        thumb_extended = self._is_thumb_extended_improved(thumb_tip, thumb_ip, thumb_mcp, wrist)
        thumb_curled = not thumb_extended
        
        # חישוב המרחק בין האגודל לאצבע המורה - סימן מובהק שהאגודל אינו חלק מאגרוף
        thumb_to_index_dist = np.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)
        index_to_middle_dist = np.sqrt((index_tip[0] - middle_tip[0])**2 + (index_tip[1] - middle_tip[1])**2)
        
        # האגודל רחוק משמעותית מהאצבע המורה ביחס למרחק בין האצבעות האחרות
        thumb_clearly_separated = thumb_to_index_dist > 2 * index_to_middle_dist
        
        # מרחק ממוצע בין קצות האצבעות
        tip_points = [index_tip, middle_tip, ring_tip, pinky_tip]
        avg_distance = 0
        count = 0
        for i in range(len(tip_points)):
            for j in range(i+1, len(tip_points)):
                dist = np.sqrt((tip_points[i][0] - tip_points[j][0])**2 + 
                              (tip_points[i][1] - tip_points[j][1])**2)
                avg_distance += dist
                count += 1
        
        if count > 0:
            avg_distance /= count
            
        # מרחק בין פיקות האצבעות (מפרקים) - למדידה יחסית
        mcp_points = [index_mcp, middle_mcp, ring_mcp, pinky_mcp]
        mcp_distance = 0
        count = 0
        for i in range(len(mcp_points)):
            for j in range(i+1, len(mcp_points)):
                dist = np.sqrt((mcp_points[i][0] - mcp_points[j][0])**2 + 
                              (mcp_points[i][1] - mcp_points[j][1])**2)
                mcp_distance += dist
                count += 1
        
        if count > 0:
            mcp_distance /= count
        
        # אגרוף מזוהה רק אם כל 5 האצבעות מכופפות (כולל הבוהן) וגם קצות האצבעות קרובים זה לזה
        # והוספת תנאי - אם האגודל בולט החוצה בצורה ברורה, זה לא אגרוף!
        fingers_close_to_each_other = avg_distance < mcp_distance * 0.8
        all_fingers_curled = index_curled and middle_curled and ring_curled and pinky_curled
        fist_detected = fingers_close_to_each_other and all_fingers_curled and thumb_curled and not thumb_clearly_separated

        # יד פתוחה - בדיקה שהאצבעות מרוחקות זו מזו וגם רובן לא מכופפות
        fingers_far_from_each_other = avg_distance > mcp_distance * 1.2
        extended_fingers_count = 4 - sum([index_curled, middle_curled, ring_curled, pinky_curled])  # מכל 4 האצבעות (ללא האגודל)
        open_palm = fingers_far_from_each_other and extended_fingers_count >= 3 and not fist_detected
        
        # עדיפות להחלטות - אגרוף מנצח כף יד פתוחה במקרה של ספק
        if fist_detected and not open_palm:
            controls['gesture_name'] = 'Brake (Fist)'
            controls['braking'] = True
            controls['throttle'] = 0.0
            controls['boost'] = False
            self._update_command_stability("BRAKE")
        elif open_palm and not fist_detected:
            controls['gesture_name'] = 'Stop (Open Palm)'
            controls['braking'] = True
            controls['throttle'] = 0.0
            controls['boost'] = False
            self._update_command_stability("STOP")
        elif thumb_extended:
            controls['gesture_name'] = 'Boost (Thumb Up)'
            controls['boost'] = True
            controls['braking'] = False
            controls['throttle'] = 1.0
            self._update_command_stability("BOOST")
        else:
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
                
        # הוספת דיבאג אינדיקציות אם במצב דיבאג
        if self.debug_mode:
            cv2.putText(frame, f"Fist: {fist_detected}", (10, 300), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(frame, f"Open palm: {open_palm}", (10, 330), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(frame, f"Fingers close: {fingers_close_to_each_other}", (10, 360), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(frame, f"Curled count: {sum([index_curled, middle_curled, ring_curled, pinky_curled])}", (10, 390), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(frame, f"Thumb extended: {thumb_extended}", (10, 420), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(frame, f"Thumb separated: {thumb_clearly_separated}", (10, 450), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
        cv2.putText(frame, f"Gesture: {controls['gesture_name']}", 
                   (frame.shape[1]//2 - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
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
    
    def _add_control_visualization(self, frame, controls):
        """Add visualization of current controls to the frame."""
        h, w, _ = frame.shape
        
        # Draw throttle bar on the right side
        bar_height = int(h * 0.6)
        bar_width = 20
        bar_x = w - 50
        bar_y = int(h * 0.2)
        
        # Draw background bar
        cv2.rectangle(frame, 
                     (bar_x, bar_y),
                     (bar_x + bar_width, bar_y + bar_height),
                     (100, 100, 100),
                     -1)
        
        # Draw throttle level
        throttle_height = int(bar_height * controls['throttle'])
        if throttle_height > 0:
            cv2.rectangle(frame,
                         (bar_x, bar_y + bar_height - throttle_height),
                         (bar_x + bar_width, bar_y + bar_height),
                         (0, 255, 0),
                         -1)
        
        # Draw calibration marks
        min_y = int(bar_y + bar_height * (1 - self.min_hand_height))
        max_y = int(bar_y + bar_height * (1 - self.max_hand_height))
        cv2.line(frame, (bar_x - 5, min_y), (bar_x + bar_width + 5, min_y), (255, 0, 0), 2)
        cv2.line(frame, (bar_x - 5, max_y), (bar_x + bar_width + 5, max_y), (0, 0, 255), 2)
        
        # Add labels
        cv2.putText(frame, "Throttle", (bar_x - 30, bar_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{controls['throttle']:.2f}", (bar_x - 30, bar_y + bar_height + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)