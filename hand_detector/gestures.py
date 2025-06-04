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
        
        # Additional parameters for speed control
        self.min_speed = 0.2  # מהירות מינימלית (20% מהמהירות המקסימלית)
        self.max_speed = 1.0  # מהירות מקסימלית
        self.no_hand_timeout = 2.0  # זמן המתנה לפני ירידה למהירות מינימלית
        self.last_hand_time = time.time()
        
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
            
            # Convert BGR to RGB BEFORE flipping for MediaPipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe BEFORE flipping
            results = self.hands.process(rgb_frame)
            
            # NOW flip the frame horizontally for display
            frame = cv2.flip(frame, 1)
            
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
                    # For drawing: flip landmarks to match flipped display frame
                    flipped_landmarks = []
                    for lm in hand_landmarks.landmark:
                        flipped_landmarks.append(type('obj', (object,), {
                            'x': 1.0 - lm.x,  # Flip x coordinate
                            'y': lm.y,        # Keep y coordinate as is
                            'z': lm.z
                        })())
                    
                    # Create flipped hand_landmarks for drawing
                    flipped_hand_landmarks = type('obj', (object,), {
                        'landmark': flipped_landmarks
                    })()
                    
                    # Draw flipped landmarks on the flipped frame
                    self.mp_draw.draw_landmarks(
                        frame,
                        flipped_hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Extract control values from ORIGINAL (non-flipped) landmarks
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
        - Throttle: hand height for speed with continuous acceleration/deceleration
        - Brake: fist gesture
        - Boost: thumb up, other fingers curled
        - Stop: open palm
        """
        # Convert landmarks to more accessible format
        h, w, c = frame.shape
        # Use normalized coordinates for gesture logic!
        norm_landmarks = [(lm.x, lm.y) for lm in landmarks.landmark]
        
        # Get key points
        wrist = norm_landmarks[0]
        thumb_tip = norm_landmarks[4]
        thumb_ip = norm_landmarks[3]  # IP joint of thumb (interphalangeal)
        index_tip = norm_landmarks[8]
        middle_tip = norm_landmarks[12]
        ring_tip = norm_landmarks[16]
        pinky_tip = norm_landmarks[20]
        
        # Get MCP (knuckle) positions for detecting finger curling
        thumb_mcp = norm_landmarks[2]
        index_mcp = norm_landmarks[5]
        middle_mcp = norm_landmarks[9]
        ring_mcp = norm_landmarks[13]
        pinky_mcp = norm_landmarks[17]
        
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
        
        # ==================== THROTTLE DETECTION WITH CONTINUOUS CONTROL ====================
        # FIXED: Use wrist position directly for throttle control
        # Get normalized hand height (0 at top of frame, 1 at bottom)
        normalized_y = wrist[1]  # Use wrist Y coordinate (already normalized by MediaPipe)
        
        # Debug: Print the actual hand position with more detail
        if self.debug_mode:
            current_time = time.time()
            if not hasattr(self, '_last_throttle_debug') or current_time - self._last_throttle_debug > 2.0:
                print(f"[THROTTLE DEBUG] Wrist Y: {normalized_y:.3f} (0=top, 1=bottom)")
                print(f"[THROTTLE DEBUG] Wrist coords: x={wrist[0]:.3f}, y={wrist[1]:.3f}")
                
                # Also check other key points for comparison
                thumb_y = thumb_tip[1]
                index_y = index_tip[1]
                print(f"[THROTTLE DEBUG] Thumb tip Y: {thumb_y:.3f}, Index tip Y: {index_y:.3f}")
                self._last_throttle_debug = current_time
        
        # Validate that we have a reasonable wrist position
        if normalized_y <= 0.01:  # If wrist is at very top (suspicious)
            print(f"⚠️ Warning: Wrist position seems invalid (y={normalized_y:.3f}), using middle finger MCP instead")
            # Fallback to middle finger MCP position
            normalized_y = middle_mcp[1] if middle_mcp[1] > 0.01 else 0.5
        
        # Initialize throttle state tracking if not exists
        if not hasattr(self, 'current_throttle'):
            self.current_throttle = 0.5
            self.last_throttle_time = time.time()
            
        # Get current time for delta calculation
        current_time = time.time()
        dt = min(0.1, current_time - self.last_throttle_time)
        self.last_throttle_time = current_time
        
        # Throttle change rates
        acceleration_rate = 0.8  # Units per second when accelerating
        deceleration_rate = 1.0  # Units per second when decelerating
        
        # Determine if hand is in upper or lower half of the frame
        if normalized_y < 0.5:  # Upper half - Accelerate
            # Calculate acceleration factor (faster acceleration when hand is higher)
            accel_factor = 1.0 - normalized_y * 2  # 1.0 at top, 0.0 at middle
            
            # Apply acceleration based on hand height
            throttle_change = acceleration_rate * dt * (0.5 + accel_factor)
            self.current_throttle += throttle_change
            
            # Debug output for significant changes
            if throttle_change > 0.05 and self.debug_mode:
                print(f"👆 Accelerating: +{throttle_change:.2f} (hand height: {normalized_y:.2f})")
                
        else:  # Lower half - Decelerate
            # Calculate deceleration factor (faster deceleration when hand is lower)
            decel_factor = (normalized_y - 0.5) * 2  # 0.0 at middle, 1.0 at bottom
            
            # Apply deceleration based on hand height
            throttle_change = deceleration_rate * dt * (0.5 + decel_factor)
            self.current_throttle -= throttle_change
            
            # Debug output for significant changes  
            if throttle_change > 0.05 and self.debug_mode:
                print(f"👇 Decelerating: -{throttle_change:.2f} (hand height: {normalized_y:.2f})")
        
        # Clamp throttle value with minimum 0.2 and maximum 1.0
        self.current_throttle = max(0.2, min(1.0, self.current_throttle))
        
        # Apply smoothing to the final throttle value
        throttle = self.prev_throttle * self.throttle_smoothing + self.current_throttle * (1 - self.throttle_smoothing)
        throttle = max(0.2, min(1.0, throttle))
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
                # Use the current (not smoothed) throttle for gesture name
                raw_throttle = self.current_throttle if hasattr(self, 'current_throttle') else controls['throttle']
                if self.debug_mode:
                    print(f"[DEBUG] Throttle (smoothed): {controls['throttle']:.2f}, Throttle (raw): {raw_throttle:.2f}")
                if raw_throttle > 0.7:
                    controls['gesture_name'] = 'High Throttle'
                    self._update_command_stability("HIGH_THROTTLE")
                elif raw_throttle < 0.3:
                    controls['gesture_name'] = 'Low Throttle'
                    self._update_command_stability("LOW_THROTTLE")
                else:
                    controls['gesture_name'] = 'Medium Throttle'
                    self._update_command_stability("MED_THROTTLE")
                
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

    def _is_thumb_extended_improved(self, thumb_tip, thumb_ip, thumb_mcp, wrist):
        """
        Improved thumb extension detection
        
        Args:
            thumb_tip: Thumb tip coordinates
            thumb_ip: Thumb IP joint coordinates  
            thumb_mcp: Thumb MCP joint coordinates
            wrist: Wrist coordinates
            
        Returns:
            bool: True if thumb is extended upward
        """
        try:
            # Check if thumb joints are in proper extended order (tip higher than IP, IP higher than MCP)
            thumb_joints_extended = thumb_tip[1] < thumb_ip[1] < thumb_mcp[1]
            
            # Calculate thumb direction vector
            thumb_vector_x = thumb_tip[0] - thumb_mcp[0]
            thumb_vector_y = thumb_tip[1] - thumb_mcp[1]
            
            # Calculate angle relative to vertical (negative Y is up)
            thumb_angle = np.degrees(np.arctan2(thumb_vector_x, -thumb_vector_y))
            
            # Thumb is extended if pointing roughly upward (within 45 degrees of vertical)
            thumb_pointing_up = abs(thumb_angle) < 45
            
            # Thumb must be significantly higher than wrist
            thumb_above_wrist = thumb_tip[1] < wrist[1] - 0.05  # Fixed: use normalized coordinates
            
            return thumb_joints_extended and thumb_pointing_up and thumb_above_wrist
            
        except Exception as e:
            print(f"⚠️ Error in thumb extension detection: {e}")
            return False
    
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
        
        # Draw steering indicator
        steering_bar_width = 200
        steering_bar_height = 20
        steering_bar_x = w // 2 - steering_bar_width // 2
        steering_bar_y = h - 50
        
        # Background
        cv2.rectangle(frame,
                     (steering_bar_x, steering_bar_y),
                     (steering_bar_x + steering_bar_width, steering_bar_y + steering_bar_height),
                     (100, 100, 100),
                     -1)
        
        # Steering position
        steering_pos = int(steering_bar_width / 2 + (controls['steering'] * steering_bar_width / 2))
        cv2.circle(frame,
                  (steering_bar_x + steering_pos, steering_bar_y + steering_bar_height // 2),
                  8,
                  (0, 0, 255),
                  -1)
        
        # Add labels
        cv2.putText(frame, "Throttle", (bar_x - 30, bar_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{controls['throttle']:.2f}", (bar_x - 30, bar_y + bar_height + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, "Steering", (steering_bar_x, steering_bar_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{controls['steering']:.2f}", (steering_bar_x, steering_bar_y + steering_bar_height + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def calculate_throttle(self, hand_landmarks, frame_height):
        """חישוב מצערת בהתבסס על גובה היד"""
        if not hand_landmarks:
            return self.min_speed  # החזר מהירות מינימלית במקום 0
        
        # חישוב גובה היד הממוצע
        hand_y = sum([lm.y for lm in hand_landmarks.landmark]) / len(hand_landmarks.landmark)
        
        # נרמול הערך (0-1, כאשר 0 = למעלה, 1 = למטה)
        normalized_y = hand_y
        
        # החזרת ערך מצערת בין min_speed למהירות מקסימלית
        # ככל שהיד נמוכה יותר - מהירות נמוכה יותר (אבל לא 0)
        throttle = self.max_speed - (normalized_y * (self.max_speed - self.min_speed))
        
        return max(self.min_speed, min(self.max_speed, throttle))