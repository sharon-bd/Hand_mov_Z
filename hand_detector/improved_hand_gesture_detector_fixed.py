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
                dist_tip = float(np.sqrt(float(dx_tip)**2 + float(dy_tip)**2))
                dist_mcp = float(np.sqrt(float(dx_mcp)**2 + float(dy_mcp)**2))
                
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
            thumb_dist_tip = float(np.sqrt(float(dx_thumb)**2 + float(dy_thumb)**2))
            thumb_dist_mcp = float(np.sqrt(float(dx_thumb_mcp)**2 + float(dy_thumb_mcp)**2))
            
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
                spacing = float(np.sqrt(float(dx)**2 + float(dy)**2))  # Safe square root calculation
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
            import traceback
            traceback.print_exc()
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
        thumb_mcp = landmark_points[2]  # בסיס האגודל

        # חישוב שיפור לזווית האגודל
        dx_thumb = float(thumb_tip[0] - wrist[0])
        dy_thumb = float(thumb_tip[1] - wrist[1])
        
        # מעבר לחישוב זווית בטווח 0-360
        thumb_angle = float(np.degrees(np.arctan2(dx_thumb, -dy_thumb)))
        if thumb_angle < 0:
            thumb_angle += 360
        
        # הגדרת טווח זוויות לא תקפות (כאשר האגודל מצביע כלפי מטה)
        INVALID_ANGLE_MIN = 80
        INVALID_ANGLE_MAX = 280
        
        # בדיקה אם האגודל מצביע בכיוון היגוי תקף
        is_valid_steering_angle = not (INVALID_ANGLE_MIN < thumb_angle < INVALID_ANGLE_MAX)

        # Store the thumb angle
        self.detection_data['thumb_angle'] = thumb_angle

        # מיפוי הזווית לערך היגוי תוך התחשבות בטווח התקף
        if is_valid_steering_angle:
            # Convert 0-360 angle to steering in [-1, 1] range
            # 0 degrees = thumb pointing up = center steering (0)
            # 0-80 degrees = right turn (0 to 1)
            # 280-360 degrees = left turn (-1 to 0)
            if 0 <= thumb_angle <= INVALID_ANGLE_MIN:
                # Thumb pointing up-right (0-80 degrees)
                raw_steering = float(thumb_angle / INVALID_ANGLE_MIN)
            elif INVALID_ANGLE_MAX <= thumb_angle <= 360:
                # Thumb pointing up-left (280-360 degrees)
                # Map 280-360 to -1..0
                raw_steering = float((thumb_angle - 360) / (360 - INVALID_ANGLE_MAX))
            else:
                # Shouldn't reach here with valid angles, but just in case
                raw_steering = 0.0
        else:
            # אם האגודל מצביע בכיוון לא תקף, שמור על ערך ההיגוי הקודם
            raw_steering = self.prev_steering

        # יישום החלקה לקבלת היגוי יציב יותר
        steering = float(self.prev_steering * self.steering_smoothing + raw_steering * (1 - self.steering_smoothing))
        steering = float(max(-1.0, min(1.0, steering)))  # הגבלת הטווח
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
            # Use different colors for valid/invalid angles
            line_color = (255, 0, 255) if is_valid_steering_angle else (0, 0, 255)
            cv2.line(frame, wrist, thumb_line_end, line_color, 2)
            cv2.putText(frame, f"Thumb Angle: {thumb_angle:.1f}", (10, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)
            cv2.putText(frame, f"Steering: {steering:.2f}", (10, 160), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Valid Angle: {is_valid_steering_angle}", (10, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)
        
        # ==================== THROTTLE DETECTION ====================
        # Detect throttle based on overall hand height (y-position)
        # Lower hand position = more throttle
        normalized_y = float(1.0 - (wrist[1] / h))  # Invert so higher hand = lower value
        raw_throttle = normalized_y
        
        # Store throttle calculation values
        self.detection_data['normalized_y'] = normalized_y
        self.detection_data['raw_throttle'] = raw_throttle
        
        # Apply non-linear mapping for better control (squared for more precision at low speeds)
        raw_throttle = float(np.power(max(0, raw_throttle), 1.5))
        
        # Apply smoothing and clamp to valid range
        throttle = float(self.prev_throttle * self.throttle_smoothing + raw_throttle * (1 - self.throttle_smoothing))
        throttle = float(max(0.0, min(1.0, throttle)))  # Clamp to valid range
        self.prev_throttle = throttle
        controls['throttle'] = throttle
        
        # Check for fist (braking gesture) - כל האצבעות מכופפות
        index_curled = index_tip[1] > index_mcp[1]  # האצבע מכופפת אם הקצה מתחת למפרק
        middle_curled = middle_tip[1] > middle_mcp[1]
        ring_curled = ring_tip[1] > ring_mcp[1]
        pinky_curled = pinky_tip[1] > pinky_mcp[1]
        thumb_curled = thumb_tip[0] > thumb_mcp[0] if wrist[0] > thumb_mcp[0] else thumb_tip[0] < thumb_mcp[0]

        # בדיקת מחוות boost (אגודל למעלה, שאר האצבעות מכופפות)
        # שימוש רק בבדיקת אגודל מורם ואצבע מורה מכופפת לזיהוי בוסט
        thumb_pointing_up = thumb_tip[1] < wrist[1] - 30  # אגודל מצביע למעלה
        boost_gesture = thumb_pointing_up and index_curled and middle_curled and ring_curled and pinky_curled

        # תיקון בדיקת אגרוף כדי שלא יכלול את מחוות האגודל למעלה
        fist_detected = index_curled and middle_curled and ring_curled and pinky_curled and thumb_curled and not boost_gesture

        # בדיקת מחוות stop sign (יד פתוחה)
        stop_sign = self._detect_stop_sign_gesture(landmark_points, frame) 

        # עדכון ערכי השליטה
        if fist_detected:
            controls['braking'] = True
            controls['gesture_name'] = 'Brake (Fist)'
        elif stop_sign:
            controls['braking'] = True
            controls['gesture_name'] = 'Stop (Open Palm)'
        elif boost_gesture:
            controls['boost'] = True
            controls['gesture_name'] = 'Boost (Thumb Up)'
        elif abs(steering) > 0.3:
            if steering < -0.3:
                controls['gesture_name'] = 'Turning Left'
            else:
                controls['gesture_name'] = 'Turning Right'
        else:
            controls['gesture_name'] = 'Forward'
        
        # Return the updated controls
        return controls
    
    def _add_minimal_visualization(self, frame, controls):
        """Add minimal visual indicators to the frame with proper text orientation."""
        # Save a copy of the original frame
        display_frame = frame.copy()
        
        h, w, _ = frame.shape
        
        # בדיקה שהאובייקט controls אינו None
        if controls is None:
            controls = {
                'gesture_name': 'Error',
                'steering': 0.0,
                'throttle': 0.0,
                'braking': False,
                'boost': False
            }
        
        # Add all text directly to the display frame (without mirroring)
        # Add gesture name (now correctly oriented)
        cv2.putText(
            display_frame,
            f"Gesture: {controls['gesture_name']}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Add steering value (now correctly oriented)
        cv2.putText(
            display_frame,
            f"Steering: {controls['steering']:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )
        
        # Add throttle value (now correctly oriented)
        cv2.putText(
            display_frame,
            f"Throttle: {controls['throttle']:.2f}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )
        
        # Add status indicators for braking and boost
        brake_color = (0, 0, 255) if controls['braking'] else (200, 200, 200)
        boost_color = (255, 165, 0) if controls['boost'] else (200, 200, 200)
        
        # Add BRAKE and BOOST text at the bottom of the screen
        cv2.putText(
            display_frame,
            "BRAKE",
            (w - 120, h - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            brake_color,
            2
        )
        
        cv2.putText(
            display_frame,
            "BOOST",
            (w - 120, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            boost_color,
            2
        )
        
        # Now flip horizontally for the mirror effect
        mirrored_frame = cv2.flip(display_frame, 1)
        
        return mirrored_frame
        
    def _create_data_panel(self, controls):
        """Create a panel with numerical data for analysis."""
        # Create a white panel
        panel_width = 500
        panel_height = 400
        panel = np.ones((panel_height, panel_width, 3), dtype=np.uint8) * 255
        
        # Draw title
        cv2.putText(panel, "Hand Gesture Detection Data", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.line(panel, (20, 40), (panel_width - 20, 40), (0, 0, 0), 1)
        
        # Draw gesture info
        cv2.putText(panel, f"Gesture: {controls['gesture_name']}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Draw numerical data
        y_pos = 110
        cv2.putText(panel, f"Thumb Angle: {self.detection_data['thumb_angle']:.1f}°", (20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_pos += 30
        
        cv2.putText(panel, f"Hand Height: {1.0 - self.detection_data['normalized_y']:.2f}", (20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_pos += 30
        
        # Draw finger status
        cv2.putText(panel, "Finger Status:", (20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_pos += 25
        
        for finger, extended in self.detection_data['finger_status'].items():
            status = "Extended" if extended else "Curled"
            color = (0, 128, 0) if extended else (0, 0, 200)
            cv2.putText(panel, f"- {finger.capitalize()}: {status}", (60, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_pos += 20
        
        # Add detection summary
        y_pos += 10
        cv2.putText(panel, "Detected Gestures:", (20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_pos += 25
        
        # Show stop sign detection
        stop_color = (0, 128, 0) if self.detection_data['stop_sign_detected'] else (0, 0, 200)
        cv2.putText(panel, f"- Stop Sign: {self.detection_data['stop_sign_detected']}", (40, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, stop_color, 1)
        y_pos += 20
        
        # Show all fingers extended
        fingers_color = (0, 128, 0) if self.detection_data['all_fingers_extended'] else (0, 0, 200)
        cv2.putText(panel, f"- All Fingers: {self.detection_data['all_fingers_extended']}", (40, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, fingers_color, 1)
        
        return panel