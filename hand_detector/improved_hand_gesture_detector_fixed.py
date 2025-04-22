import cv2
import mediapipe as mp
import numpy as np

class EnhancedHandGestureDetector:
    """Enhanced class to detect hand gestures and convert them to car control signals."""
    
    def __init__(self):
        """Initialize the hand gesture detector with MediaPipe."""
        self.mp_hands = mp.solutions.hands
        # Define image dimensions
        self.image_width = 640  # Default width
        self.image_height = 480  # Default height
        
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
            # Update image dimensions based on the current frame
            h, w, _ = frame.shape
            self.image_width = w
            self.image_height = h

            # First create a mirrored copy of the input frame
            mirrored_frame = cv2.flip(frame.copy(), 1)
            
            # Dimensions for debug info
            h, w, _ = mirrored_frame.shape
            
            # Convert BGR to RGB for MediaPipe processing
            rgb_frame = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2RGB)
            
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
                    # Draw hand landmarks on the mirrored frame
                    self.mp_draw.draw_landmarks(
                        mirrored_frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Extract control values from hand landmarks
                    controls = self._extract_controls_from_landmarks(hand_landmarks, mirrored_frame, controls)
            else:
                # Reset stability counter when no hand detected
                self.command_stability_count = 0
            
            # Add visual indicators to the frame with proper text orientation
            processed_frame = self._add_visualization_to_mirrored_frame(mirrored_frame, controls)
            
            # Create separate data panel
            data_panel = self._create_data_panel(controls)
            
            # Add speed and direction mappings for compatibility with the Car class
            controls['speed'] = controls['throttle']
            controls['direction'] = controls['steering']
            
            return controls, processed_frame, data_panel
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

        # בדיקה מדויקת יותר לזיהוי אגודל מורם (Thumb Up)
        # תנאים: אגודל זקוף, גבוה מהאצבעות, רחוק מהאצבע המורה, שאר האצבעות מכופפות
        thumb_tip = landmark_points[4]
        thumb_ip = landmark_points[3]
        thumb_mcp = landmark_points[2]
        wrist = landmark_points[0]
        index_tip = landmark_points[8]
        middle_tip = landmark_points[12]
        ring_tip = landmark_points[16]
        pinky_tip = landmark_points[20]

        # מרחקים
        thumb_tip_to_ip = np.linalg.norm(np.array(thumb_tip) - np.array(thumb_ip))
        thumb_ip_to_mcp = np.linalg.norm(np.array(thumb_ip) - np.array(thumb_mcp))
        thumb_tip_to_index_tip = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))
        thumb_tip_to_wrist = np.linalg.norm(np.array(thumb_tip) - np.array(wrist))

        # יחס זקיפות האגודל
        distance_ratio = thumb_tip_to_ip / thumb_ip_to_mcp if thumb_ip_to_mcp > 0 else 0

        # חישוב זווית בין שלושת מפרקי האגודל (MCP, IP, TIP)
        def angle_between_points(a, b, c):
            # זווית ב-b בין a-b-c
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)
            ba = a - b
            bc = c - b
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
            return angle

        thumb_angle = angle_between_points(thumb_mcp, thumb_ip, thumb_tip)

        # האגודל גבוה מכל שאר האצבעות (y קטן יותר)
        thumb_higher_than_fingers = (
            thumb_tip[1] < min(index_tip[1], middle_tip[1], ring_tip[1], pinky_tip[1]) - 20
        )

        # האגודל רחוק מהאצבע המורה (לא צמוד)
        thumb_far_from_index = thumb_tip_to_index_tip > 0.7 * thumb_tip_to_wrist

        # אגודל מקופל אם היחס קטן, קצה קרוב למפרק, או זווית חדה
        thumb_curled = (
            distance_ratio < 0.7 or
            thumb_tip_to_ip < thumb_ip_to_mcp * 0.8 or
            thumb_angle < 120  # זווית חדה = אגודל מקופל
        )

        # בדיקה נוספת: האם האגודל קרוב לאצבע המורה (מאפיין אגרוף)
        thumb_close_to_index = thumb_tip_to_index_tip < 0.4 * thumb_tip_to_wrist  # מרחק קטן = אגודל קרוב לאצבע

        # עדכון ההגדרה של אגרוף
        fist_detected = index_curled and middle_curled and ring_curled and pinky_curled and thumb_curled and thumb_close_to_index

        # הוספת מידע דיבאג
        if self.debug_mode:
            # Rearranged and spaced out debug text lines - at least 20px apart
            cv2.putText(frame, f"ThumbUp: {thumb_pointing_up}", (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            cv2.putText(frame, f"Thumb>Fingers: {thumb_higher_than_fingers}", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            cv2.putText(frame, f"Thumb-Index dist: {thumb_tip_to_index_tip:.1f}", (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            # Fixed overlapping lines - changed y-position from 440 to 445 for "Fist" text
            cv2.putText(frame, f"Fist: {fist_detected}", (10, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            # Move "Thumb angle" text below "Fist" text
            cv2.putText(frame, f"Thumb angle: {thumb_angle:.1f}", (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            # Move "ThumbCloseToIndex" text down
            cv2.putText(frame, f"ThumbCloseToIndex: {thumb_close_to_index}", (10, 485), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            # Empty line not needed since we've properly spaced everything
            cv2.putText(frame, f"ThumbUpward: {thumb_pointing_upward}", (10, 505), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            cv2.putText(frame, f"ThumbToWristAngle: {thumb_to_wrist_angle:.1f}", (10, 525), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

        boost_gesture = thumb_pointing_up and index_curled and middle_curled and ring_curled and pinky_curled

        # תנאי Thumb Up: אגודל זקוף, גבוה, רחוק מהאצבע, שאר האצבעות מכופפות, זווית קהה
        # בדיקה נוספת: האם האגודל מצביע כלפי מעלה ביחס לשורש כף היד
        thumb_to_wrist_dx = thumb_tip[0] - wrist[0]
        thumb_to_wrist_dy = thumb_tip[1] - wrist[1]
        thumb_to_wrist_angle = np.degrees(np.arctan2(-thumb_to_wrist_dy, thumb_to_wrist_dx))  # -dy כי ציר Y הפוך
        if thumb_to_wrist_angle < 0:
            thumb_to_wrist_angle += 360

        # האגודל צריך להצביע כלפי מעלה (זווית בין 270 ל-90 מעלות דרך 0)
        thumb_pointing_upward = (270 <= thumb_to_wrist_angle <= 360) or (0 <= thumb_to_wrist_angle <= 90)

        thumb_pointing_up = (
            distance_ratio > 0.8 and
            thumb_higher_than_fingers and
            thumb_far_from_index and
            thumb_angle > 150 and  # זווית קהה = אגודל זקוף
            thumb_pointing_upward  # חדש: האגודל מצביע כלפי מעלה
        )

        if self.debug_mode:
            cv2.putText(frame, f"ThumbUp: {thumb_pointing_up}", (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            cv2.putText(frame, f"Thumb>Fingers: {thumb_higher_than_fingers}", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            cv2.putText(frame, f"Thumb-Index dist: {thumb_tip_to_index_tip:.1f}", (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            cv2.putText(frame, f"Thumb angle: {thumb_angle:.1f}", (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            cv2.putText(frame, f"ThumbUpward: {thumb_pointing_upward}", (10, 505), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            cv2.putText(frame, f"ThumbToWristAngle: {thumb_to_wrist_angle:.1f}", (10, 525), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

        boost_gesture = thumb_pointing_up and index_curled and middle_curled and ring_curled and pinky_curled

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
    
    def _add_visualization_to_mirrored_frame(self, frame, controls):
        """Add visualization to the mirrored frame with correct text orientation and positioning."""
        h, w, _ = frame.shape

        # Increase margin at the bottom to prevent cutting off text
        margin = 120  # was 80, now 120 to avoid cutting text
        frame_with_margin = np.zeros((h + margin, w, 3), dtype=np.uint8)
        frame_with_margin[:h, :] = frame

        # Add gesture name at the top-left of the frame
        cv2.putText(frame_with_margin, f"Gesture: {controls['gesture_name']}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # Add steering and throttle values below the gesture name
        cv2.putText(frame_with_margin, f"Steering: {controls['steering']:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame_with_margin, f"Throttle: {controls['throttle']:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        brake_color = (0, 0, 255) if controls['braking'] else (200, 200, 200)
        boost_color = (255, 165, 0) if controls['boost'] else (200, 200, 200)
        # Move up by 120px to avoid being cut off
        bottom_y_position = h - 80 + 40  # shift up by 40px relative to previous
        cv2.putText(frame_with_margin, "BRAKE", (w - 120, bottom_y_position),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, brake_color, 2)
        cv2.putText(frame_with_margin, "BOOST", (w - 120, bottom_y_position + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, boost_color, 2)
        # Move FPS to top-right
        if hasattr(self, 'fps'):
            cv2.putText(frame_with_margin, f"FPS: {self.fps:.1f}", (w - 150, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return frame_with_margin

    def _create_data_panel(self, controls):
        """Create a panel with numerical data for analysis."""
        # Create a white panel
        panel_width = 500
        panel_height = 400
        panel = np.ones((panel_height, panel_width, 3), dtype=np.uint8) * 255
        
        # Draw title and basic info
        cv2.putText(panel, "Hand Gesture Detection Data", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.line(panel, (20, 40), (panel_width - 20, 40), (0, 0, 0), 1)
        
        # Safely access detection data with get() method
        cv2.putText(panel, f"Gesture: {controls['gesture_name']}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        y_pos = 110
        cv2.putText(panel, f"Thumb Angle: {self.detection_data.get('thumb_angle', 0):.1f}°", 
                    (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_pos += 30
        
        cv2.putText(panel, f"Hand Height: {1.0 - self.detection_data.get('normalized_y', 0):.2f}", 
                    (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_pos += 30
        
        # Draw finger status with safe dictionary access
        cv2.putText(panel, "Finger Status:", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y_pos += 25
        
        for finger, extended in self.detection_data.get('finger_status', {}).items():
            status = "Extended" if extended else "Curled"
            color = (0, 128, 0) if extended else (0, 0, 200)
            cv2.putText(panel, f"- {finger.capitalize()}: {status}", (60, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_pos += 20
        
        return panel

    def _flip_landmarks_horizontally(self, landmarks, frame_width):
        """Flip hand landmarks horizontally for mirrored display."""
        # Create a copy of the landmarks
        flipped_landmarks = mp.solutions.hands.HandLandmark()
        
        # Copy all attributes from the original landmarks
        for i in range(21):  # 21 hand landmarks
            # Get the original landmark
            landmark = landmarks.landmark[i]
            
            # Create a new landmark with flipped x-coordinate
            new_landmark = type(landmark)()
            new_landmark.x = 1.0 - landmark.x  # Flip horizontally (0.0-1.0 range)
            new_landmark.y = landmark.y        # Keep y-coordinate the same
            new_landmark.z = landmark.z        # Keep z-coordinate the same
            
            # Set the new landmark
            flipped_landmarks.landmark[i] = new_landmark
        
        return flipped_landmarks

    def _check_thumb_extended(self, thumb_tip, thumb_ip, thumb_mcp, wrist):
        """
        Unified method to check if thumb is extended.
        Returns True if thumb is extended, False if curled.
        """
        # Calculate distances
        thumb_tip_to_ip = np.linalg.norm(np.array(thumb_tip) - np.array(thumb_ip))
        thumb_ip_to_mcp = np.linalg.norm(np.array(thumb_ip) - np.array(thumb_mcp))
        thumb_tip_to_wrist = np.linalg.norm(np.array(thumb_tip) - np.array(wrist))  # <-- fixed parenthesis
        
        # Calculate ratio for extension check
        distance_ratio = thumb_tip_to_ip / thumb_ip_to_mcp if thumb_ip_to_mcp > 0 else 0
        
        # Calculate angle between segments
        def angle_between_points(a, b, c):
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)
            ba = a - b
            bc = c - b
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
            return angle

        thumb_angle = angle_between_points(thumb_mcp, thumb_ip, thumb_tip)
        
        # Check if thumb is pointing upward
        thumb_to_wrist_dx = thumb_tip[0] - wrist[0]
        thumb_to_wrist_dy = thumb_tip[1] - wrist[1]
        thumb_to_wrist_angle = np.degrees(np.arctan2(-thumb_to_wrist_dy, thumb_to_wrist_dx))
        if thumb_to_wrist_angle < 0:
            thumb_to_wrist_angle += 360
            
        thumb_pointing_upward = (270 <= thumb_to_wrist_angle <= 360) or (0 <= thumb_to_wrist_angle <= 90)
        
        # Determine if thumb is extended based on multiple criteria
        thumb_extended = (
            distance_ratio > 0.7 and
            thumb_angle > 120 and   # Extended thumb has obtuse angle
            thumb_pointing_upward    # Thumb should point upward to be considered extended
        )
        
        return thumb_extended