import cv2
import mediapipe as mp
import numpy as np

class EnhancedHandGestureDetector:
    """Enhanced class to detect hand gestures and convert them to car control signals."""
    
    def __init__(self):
        """Initialize the hand gesture detector with MediaPipe."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Track only one hand for simplicity
            min_detection_confidence=0.6,  # Increased from 0.4 for more reliable detection
            min_tracking_confidence=0.5    # Increased from 0.4 for better tracking
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
        
        # Store all detection data for external display
        self.detection_data = {}
        
    def detect_gestures(self, frame):
        """
        Detect hand gestures in the given frame and return control signals.
        
        Args:
            frame: CV2 image frame
            
        Returns:
            controls: Dictionary with control values (steering, throttle, braking, boost)
            processed_frame: Frame with visualization of detected hands and controls
            data_panel: Additional image panel with numerical data
        """
        try:
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
                'gesture_name': 'No hand detected',
                'speed': 0.0,        # For compatibility with Car.update()
                'direction': 0.0     # For compatibility with Car.update()
            }
            
            # Reset detection data
            self.detection_data = {
                'thumb_angle': 0.0,
                'hand_position_y': 0,
                'normalized_y': 0.0,
                'raw_throttle': 0.0,
                'all_fingers_extended': False,
                'fist_detected': False,
                'stop_sign_detected': False,
                'finger_status': {
                    'thumb': False,
                    'index': False,
                    'middle': False,
                    'ring': False,
                    'pinky': False
                }
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
            
            # Add minimal visualization to the camera frame
            minimal_frame = self._add_minimal_visualization(frame.copy(), controls)
            
            # Create separate data panel
            data_panel = self._create_data_panel(controls)
            
            # Add speed and direction mappings for compatibility with the Car class
            controls['speed'] = controls['throttle']
            controls['direction'] = controls['steering']
            
            return controls, minimal_frame, data_panel
        except Exception as e:
            print(f"Error in gesture detection: {e}")
            return {
                'steering': 0.0,
                'throttle': 0.0,
                'braking': False,
                'boost': False,
                'gesture_name': 'Error',
                'speed': 0.0,
                'direction': 0.0
            }, frame, self._create_error_panel()
    
    def _create_error_panel(self):
        """Create an error panel when detection fails"""
        panel = np.ones((300, 400, 3), dtype=np.uint8) * 255
        cv2.putText(panel, "ERROR IN HAND DETECTION", (30, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return panel

    def _detect_stop_sign_gesture(self, landmark_points, frame):
        """
        זיהוי מחוות STOP (יד פתוחה כמו תמרור עצור).
        מחזיר True אם המחווה זוהתה, False אחרת.
        """
        try:
            h, w, _ = frame.shape
            
            # נקודות מפתח
            wrist = landmark_points[0]
            thumb_tip = landmark_points[4]
            index_tip = landmark_points[8]
            middle_tip = landmark_points[12]
            ring_tip = landmark_points[16]
            pinky_tip = landmark_points[20]
            
            # בסיסי האצבעות (מפרקים)
            thumb_mcp = landmark_points[2]
            index_mcp = landmark_points[5]
            middle_mcp = landmark_points[9]
            ring_mcp = landmark_points[13]
            pinky_mcp = landmark_points[17]
            
            # 1. בדיקה שכל האצבעות פרושות (לא מכופפות)
            # האצבע נחשבת פרושה אם קצה האצבע נמצא רחוק יותר מהמפרק ביחס לשורש כף היד
            
            # 1.1 עבור האצבע המורה, האמצעית, הטבעת והזרת
            finger_extended = []
            finger_names = ["index", "middle", "ring", "pinky"]
            
            for i, (tip, mcp, name) in enumerate(zip(
                [index_tip, middle_tip, ring_tip, pinky_tip], 
                [index_mcp, middle_mcp, ring_mcp, pinky_mcp],
                finger_names
            )):
                # חישוב בטוח של מרחק - explicit numeric calculation
                dx_tip = float(tip[0] - wrist[0])
                dy_tip = float(tip[1] - wrist[1])
                dx_mcp = float(mcp[0] - wrist[0])
                dy_mcp = float(mcp[1] - wrist[1])
                
                # Fixed calculation using explicit square root for Euclidean distance
                dist_tip = np.sqrt(dx_tip**2 + dy_tip**2)
                dist_mcp = np.sqrt(dx_mcp**2 + dy_mcp**2)
                
                # האצבע פרושה אם הקצה רחוק יותר מהמפרק
                is_extended = False
                if dist_mcp > 0:  # מניעת חלוקה באפס
                    is_extended = dist_tip > dist_mcp * 1.2
                
                finger_extended.append(is_extended)
                self.detection_data['finger_status'][name] = is_extended
            
            # 1.2 עבור האגודל (מקרה מיוחד)
            dx_thumb = float(thumb_tip[0] - wrist[0])
            dy_thumb = float(thumb_tip[1] - wrist[1])
            dx_thumb_mcp = float(thumb_mcp[0] - wrist[0])
            dy_thumb_mcp = float(thumb_mcp[1] - wrist[1])
            
            # Fixed calculation using explicit square root
            thumb_dist_tip = np.sqrt(dx_thumb**2 + dy_thumb**2)
            thumb_dist_mcp = np.sqrt(dx_thumb_mcp**2 + dy_thumb_mcp**2)
            
            thumb_extended = False
            if thumb_dist_mcp > 0:
                thumb_extended = thumb_dist_tip > thumb_dist_mcp
            
            self.detection_data['finger_status']["thumb"] = thumb_extended
            
            # 2. בדיקה שהיד פתוחה ומורמת
            all_fingers_extended = all(finger_extended) and thumb_extended
            self.detection_data['all_fingers_extended'] = all_fingers_extended
            
            # 3. בדיקה שהאצבעות פרושות במרחק סביר אחת מהשנייה
            # מחשב מרחקים בין האצבעות הסמוכות
            finger_tips = [index_tip, middle_tip, ring_tip, pinky_tip]
            finger_spacings = []
            
            for i in range(len(finger_tips) - 1):
                dx = float(finger_tips[i][0] - finger_tips[i+1][0])
                dy = float(finger_tips[i][1] - finger_tips[i+1][1])
                spacing = np.sqrt(dx**2 + dy**2)  # Safe square root calculation
                finger_spacings.append(spacing)
            
            # האצבעות צריכות להיות במרחק דומה זו מזו
            fingers_evenly_spaced = True
            if len(finger_spacings) > 1:
                min_spacing = min(finger_spacings)
                max_spacing = max(finger_spacings)
                if min_spacing > 0:  # מניעת חלוקה באפס
                    fingers_evenly_spaced = (max_spacing < min_spacing * 2.0)  # רווחים פחות או יותר שווים
            
            # הוספת מידע דיבאג אם נדרש
            if self.debug_mode:
                cv2.putText(frame, f"All Fingers Extended: {all_fingers_extended}", 
                           (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"Fingers Spaced: {fingers_evenly_spaced}", 
                           (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # הוספת מידע על פרישת כל אצבע
                fingers = ["Index", "Middle", "Ring", "Pinky", "Thumb"]
                extensions = finger_extended + [thumb_extended]
                for i, (finger, extended) in enumerate(zip(fingers, extensions)):
                    color = (0, 255, 0) if extended else (0, 0, 255)
                    cv2.putText(frame, f"{finger}: {extended}", 
                               (w - 150, 30 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # מחוות STOP זוהתה אם כל האצבעות פרושות ובמרווחים סבירים
            stop_gesture_detected = all_fingers_extended and fingers_evenly_spaced
            self.detection_data['stop_sign_detected'] = stop_gesture_detected
            
            return stop_gesture_detected
        
        except Exception as e:
            print(f"Error in _detect_stop_sign_gesture: {e}")
            return False  # במקרה של שגיאה, לא מזהים את המחווה

    def _extract_controls_from_landmarks(self, landmarks, frame, controls):
        """
        Extract control values from hand landmarks with improved detection.
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
        
        # Store hand position data
        self.detection_data['hand_position_y'] = wrist[1]
        
        # ==================== STEERING DETECTION ====================
        # מדידת זווית האגודל ביחס למרכז כף היד
        thumb_tip = landmark_points[4]  # קצה האגודל
        wrist = landmark_points[0]      # שורש כף היד

        # חישוב הזווית של האגודל ביחס לציר האנכי - ensure all values are floats
        dx_thumb = float(thumb_tip[0] - wrist[0])
        dy_thumb = float(thumb_tip[1] - wrist[1])
        thumb_angle = np.degrees(np.arctan2(dx_thumb, -dy_thumb))  # שימוש בהיפוך הציר האנכי

        # Store the thumb angle
        self.detection_data['thumb_angle'] = thumb_angle

        # מיפוי הזווית לערך היגוי
        # אגודל ישר למעלה (0 מעלות) = היגוי במרכז (0)
        # אגודל מוטה ימינה (90 מעלות) = היגוי מלא ימינה (1.0)
        # אגודל מוטה שמאלה (-90 מעלות) = היגוי מלא שמאלה (-1.0)

        if -90 <= thumb_angle <= 90:
            # נירמול לטווח של -1 עד 1
            raw_steering = thumb_angle / 90.0
        else:
            # אם האגודל מוטה יותר מדי, השתמש בערכים מקסימליים
            raw_steering = -1.0 if thumb_angle < -90 else 1.0

        # יישום החלקה לקבלת היגוי יציב יותר
        steering = self.prev_steering * self.steering_smoothing + raw_steering * (1 - self.steering_smoothing)
        steering = max(-1.0, min(1.0, steering))  # הגבלת הטווח
        self.prev_steering = steering
        controls['steering'] = steering

        # הוספת קווי חיווי עבור זווית האגודל אם מצב דיבאג מופעל
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
        # Detect throttle based on overall hand height (y-position)
        # Lower hand position = more throttle
        normalized_y = 1.0 - (wrist[1] / h)  # Invert so higher hand = lower value
        raw_throttle = normalized_y
        
        # Store throttle calculation values
        self.detection_data['normalized_y'] = normalized_y
        self.detection_data['raw_throttle'] = raw_throttle
        
        # Apply non-linear mapping for better control (squared for more precision at low speeds)
        raw_throttle = raw_throttle ** 1.5  # Exponential curve gives more control
        
        # Apply smoothing and clamp to valid range
        throttle = self.prev_throttle * self.throttle_smoothing + raw_throttle * (1 - self.throttle_smoothing)
        throttle = max(0.0, min(1.0, throttle))  # Clamp to valid range
        self.prev_throttle = throttle
        controls['throttle'] = throttle
        
        # ==================== GESTURE DETECTION ====================
        # בדיקה אם האצבעות מכופפות
        index_curled = index_tip[1] > index_mcp[1]
        middle_curled = middle_tip[1] > middle_mcp[1]
        ring_curled = ring_tip[1] > ring_mcp[1]
        pinky_curled = pinky_tip[1] > pinky_mcp[1]
        thumb_curled = thumb_tip[0] > thumb_mcp[0] if wrist[0] > thumb_mcp[0] else thumb_tip[0] < thumb_mcp[0]

        # זיהוי אגרוף (כל האצבעות מכופפות, כולל האגודל)
        fist_detected = index_curled and middle_curled and ring_curled and pinky_curled and thumb_curled
        self.detection_data['fist_detected'] = fist_detected

        # בדיקה אם האצבעות מורמות (לא מכופפות)
        fingers_extended = (
            not index_curled and
            not middle_curled and
            not ring_curled and
            not pinky_curled
        )

        # קריאה לפונקציה החדשה לזיהוי מחוות עצור
        stop_sign_gesture = self._detect_stop_sign_gesture(landmark_points, frame)

        # הוספת ויזואליזציה של נתוני הזיהוי במצב דיבוג
        if self.debug_mode:
            cv2.putText(frame, f"Fingers Extended: {fingers_extended}", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Fist Detected: {fist_detected}", (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # שינוי: BOOST רק במצב אגרוף וכאשר מיקום היד גבוה
        boost_gesture = fist_detected and wrist[1] < h/2

        # זיהוי מחוות בלימה (אגרוף רגיל)
        brake_gesture = fist_detected and wrist[1] >= h/2

        # קביעת פקודות בהתאם למחוות שזוהו
        if stop_sign_gesture:
            controls['gesture_name'] = 'Stop (Traffic Sign)'
            controls['braking'] = True  # עצירת חירום עם מחוות עצור
            controls['throttle'] = 0.0
            controls['boost'] = False
            self._update_command_stability("STOP")
        elif boost_gesture:
            controls['gesture_name'] = 'Boost'
            controls['boost'] = True
            controls['braking'] = False
            controls['throttle'] = 1.0  # גז מלא במצב BOOST
            self._update_command_stability("FORWARD_BOOST")
        elif brake_gesture:
            controls['gesture_name'] = 'Brake'
            controls['braking'] = True
            controls['boost'] = False
            controls['throttle'] = 0.0  # אין גז במצב בלימה
            self._update_command_stability("STOP")
        else:
            # נהיגה רגילה עם היגוי וגז
            controls['braking'] = False
            controls['boost'] = False
            
            if abs(steering) > 0.3:  # היגוי משמעותי
                if steering < -0.3:
                    controls['gesture_name'] = 'Turning Left'
                    self._update_command_stability("LEFT")
                else:
                    controls['gesture_name'] = 'Turning Right'
                    self._update_command_stability("RIGHT")
            else:
                controls['gesture_name'] = 'Forward'
                self._update_command_stability("FORWARD")
                
        return controls
        
    def _update_command_stability(self, command):
        """Track command stability to avoid jitter in command sending."""
        if command == self.last_command:
            self.command_stability_count += 1
        else:
            self.last_command = command
            self.command_stability_count = 1
            
    def get_stable_command(self):
        """Get the current command only if it's stable enough."""
        if self.command_stability_count >= self.stability_threshold:
            return self.last_command
        return None
    
    def _add_minimal_visualization(self, frame, controls):
        """
        Add minimal visualization to the camera frame.
        Only show essential indicators and the current gesture.
        """
        # Display current gesture name at the top of the frame
        cv2.putText(
            frame, 
            f"Gesture: {controls['gesture_name']}", 
            (frame.shape[1]//2 - 100, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        # If stop gesture detected, add a bold STOP indicator
        if controls['gesture_name'] == 'Stop (Traffic Sign)':
            cv2.putText(
                frame,
                "STOP SIGN DETECTED",
                (frame.shape[1]//2 - 150, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            
        return frame

    def _create_data_panel(self, controls):
        """
        Create a separate panel with all detection data and numerical values.
        """
        # Create a white panel
        panel_width = 400
        panel_height = 450
        panel = np.ones((panel_height, panel_width, 3), dtype=np.uint8) * 255
        
        # Add title
        cv2.putText(panel, "Hand Gesture Detection Data", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.line(panel, (20, 40), (panel_width - 20, 40), (0, 0, 0), 1)
        
        # Add gesture section
        y_pos = 80
        cv2.putText(panel, "CURRENT GESTURE", (20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y_pos += 30
        cv2.putText(panel, f"Detected: {controls['gesture_name']}", (30, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        y_pos += 25
        cv2.putText(panel, f"Stability: {self.command_stability_count}/{self.stability_threshold}", (30, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_pos += 25
        if self.get_stable_command():
            cv2.putText(panel, f"Stable Command: {self.get_stable_command()}", (30, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 102, 0), 1)
        else:
            cv2.putText(panel, "Stable Command: None", (30, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        
        # Add control values section
        y_pos += 40
        cv2.putText(panel, "CONTROL VALUES", (20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y_pos += 30
        
        # Steering value with bar visualization
        cv2.putText(panel, f"Steering: {controls['steering']:.2f}", (30, y_pos), 
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
        # Throttle value with bar visualization
        cv2.putText(panel, f"Throttle: {controls['throttle']:.2f}", (30, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Draw throttle bar
        cv2.rectangle(panel, (bar_x, y_pos - 12), (bar_x + bar_width, y_pos + 3), (220, 220, 220), -1)
        cv2.rectangle(panel, (bar_x, y_pos - 12), (bar_x + bar_width, y_pos + 3), (0, 0, 0), 1)
        
        # Fill bar based on throttle value
        fill_width = int(bar_width * controls['throttle'])
        cv2.rectangle(panel, (bar_x, y_pos - 12), (bar_x + fill_width, y_pos + 3), (0, 255, 0), -1)
        
        y_pos += 30
        # Brake and boost status
        brake_color = (0, 0, 255) if controls['braking'] else (150, 150, 150)
        boost_color = (255, 165, 0) if controls['boost'] else (150, 150, 150)
        
        cv2.putText(panel, "Braking:", (30, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.circle(panel, (120, y_pos - 5), 8, brake_color, -1)
        
        cv2.putText(panel, "Boost:", (200, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.circle(panel, (260, y_pos - 5), 8, boost_color, -1)
        
        # Add raw detection data section
        y_pos += 40
        cv2.putText(panel, "RAW DETECTION DATA", (20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y_pos += 30
        
        cv2.putText(panel, f"Thumb Angle: {self.detection_data['thumb_angle']:.2f}°", (30, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_pos += 25
        
        cv2.putText(panel, f"Hand Position Y: {self.detection_data['hand_position_y']}", (30, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_pos += 25
        
        cv2.putText(panel, f"Normalized Y: {self.detection_data['normalized_y']:.2f}", (30, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_pos += 25
        
        cv2.putText(panel, f"Raw Throttle: {self.detection_data['raw_throttle']:.2f}", (30, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_pos += 25
        
        # Add finger status section
        y_pos += 15
        cv2.putText(panel, "FINGER STATUS", (20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y_pos += 30
        
        fingers = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        x_pos = 30
        
        for finger in fingers:
            is_extended = self.detection_data['finger_status'].get(finger.lower(), False)
            status_color = (0, 128, 0) if is_extended else (128, 0, 0)
            status_text = "Extended" if is_extended else "Curled"
            
            cv2.putText(panel, f"{finger}: {status_text}", (x_pos, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            
            if finger == "Middle":  # Line break after middle finger
                y_pos += 25
                x_pos = 30
            else:
                x_pos += 150
                
        # Add gesture detection results
        y_pos += 25
        cv2.putText(panel, f"Fist Detected: {self.detection_data['fist_detected']}", (30, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_pos += 25
        
        cv2.putText(panel, f"All Fingers Extended: {self.detection_data['all_fingers_extended']}", (30, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_pos += 25
        
        stop_color = (0, 0, 255) if self.detection_data['stop_sign_detected'] else (0, 0, 0)
        cv2.putText(panel, f"Stop Sign Detected: {self.detection_data['stop_sign_detected']}", (30, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, stop_color, 1)
        
        return panel
    
    def _add_control_visualization(self, frame, controls):
        """Add visual indicators of the current controls to the frame."""
        h, w, _ = frame.shape
        
        # Draw background panel for controls
        panel_height = 120
        panel_y = h - panel_height - 10
        panel_width = 250
        cv2.rectangle(frame, (10, panel_y), (panel_width + 10, h - 10), (230, 230, 230), -1)
        cv2.rectangle(frame, (10, panel_y), (panel_width + 10, h - 10), (0, 0, 0), 1)
        
        # Draw steering indicator
        steering = controls['steering']
        cv2.putText(frame, "Steering:", (20, panel_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        steer_center_x = 130
        steer_width = 100
        steer_y = panel_y + 30
        
        # Steering box
        cv2.rectangle(frame, 
                      (steer_center_x - steer_width//2, steer_y - 15), 
                      (steer_center_x + steer_width//2, steer_y + 15), 
                      (200, 200, 200), -1)
        cv2.rectangle(frame, 
                      (steer_center_x - steer_width//2, steer_y - 15), 
                      (steer_center_x + steer_width//2, steer_y + 15), 
                      (0, 0, 0), 1)
        
        # Steering indicator
        steer_pos = int(steer_center_x + steering * steer_width/2)
        cv2.circle(frame, (steer_pos, steer_y), 10, (0, 0, 255), -1)
        
        # Draw throttle indicator
        throttle = controls['throttle']
        cv2.putText(frame, "Throttle:", (20, panel_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        throttle_x = 130
        throttle_height = 50
        throttle_width = 30
        throttle_y = panel_y + 50
        
        # Throttle box
        cv2.rectangle(frame, 
                     (throttle_x, throttle_y), 
                     (throttle_x + throttle_width, throttle_y + throttle_height), 
                     (200, 200, 200), -1)
        cv2.rectangle(frame, 
                     (throttle_x, throttle_y), 
                     (throttle_x + throttle_width, throttle_y + throttle_height), 
                     (0, 0, 0), 1)
        
        # Throttle fill
        filled_height = int(throttle_height * throttle)
        cv2.rectangle(frame, 
                     (throttle_x, throttle_y + throttle_height - filled_height), 
                     (throttle_x + throttle_width, throttle_y + throttle_height), 
                     (0, 255, 0), -1)
        
        # Draw brake and boost indicators
        brake_color = (0, 0, 255) if controls['braking'] else (200, 200, 200)
        cv2.circle(frame, (50, panel_y + 110), 15, brake_color, -1)
        cv2.putText(frame, "Brake", (30, panel_y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, brake_color, 2)
        
        boost_color = (255, 165, 0) if controls['boost'] else (200, 200, 200)
        cv2.circle(frame, (120, panel_y + 110), 15, boost_color, -1)
        cv2.putText(frame, "Boost", (100, panel_y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, boost_color, 2)
        
        # Add stability indicator
        stability_x = panel_width - 40
        cv2.putText(frame, f"Stability: {self.command_stability_count}/{self.stability_threshold}", 
                    (stability_x - 80, panel_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Return the frame with added visualizations
        return frame
