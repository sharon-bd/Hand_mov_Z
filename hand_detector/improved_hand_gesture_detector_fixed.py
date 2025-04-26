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
            finger_extended = []
            finger_names = ["index", "middle", "ring", "pinky"]
            
            for i, (tip, mcp, name) in enumerate(zip(
                [index_tip, middle_tip, ring_tip, pinky_tip], 
                [index_mcp, middle_mcp, ring_mcp, pinky_mcp],
                finger_names
            )):
                dx_tip = float(tip[0] - wrist[0])
                dy_tip = float(tip[1] - wrist[1])
                dx_mcp = float(mcp[0] - wrist[0])
                dy_mcp = float(mcp[1] - wrist[1])
                
                dist_tip = float(np.sqrt(float(dx_tip)**2 + float(dy_tip)**2))
                dist_mcp = float(np.sqrt(float(dx_mcp)**2 + float(dy_mcp)**2))
                
                is_extended = False
                if dist_mcp > 0:
                    is_extended = dist_tip > dist_mcp * 1.2
                
                finger_extended.append(is_extended)
                self.detection_data['finger_status'][name] = is_extended
            
            # 1.2 עבור האגודל
            thumb_ip = landmark_points[3]  # נקודת המפרק האמצעי של האגודל
            thumb_extended = self._check_thumb_extended(thumb_tip, thumb_ip, thumb_mcp, wrist, context="open_palm")
            self.detection_data['finger_status']["thumb"] = thumb_extended
            
            # 2. בדיקה שהיד פתוחה ומורמת
            all_fingers_extended = all(finger_extended) and thumb_extended
            self.detection_data['all_fingers_extended'] = all_fingers_extended
            
            # 3. בדיקה שהאצבעות פרושות במרחק סביר אחת מהשנייה
            finger_tips = [index_tip, middle_tip, ring_tip, pinky_tip]
            finger_spacings = []
            
            for i in range(len(finger_tips) - 1):
                dx = float(finger_tips[i][0] - finger_tips[i+1][0])
                dy = float(finger_tips[i][1] - finger_tips[i+1][1])
                spacing = float(np.sqrt(float(dx)**2 + float(dy)**2))
                finger_spacings.append(spacing)
            
            fingers_evenly_spaced = True
            if len(finger_spacings) > 1:
                min_spacing = min(finger_spacings)
                max_spacing = max(finger_spacings)
                if min_spacing > 0:
                    fingers_evenly_spaced = (max_spacing < min_spacing * 2.0)
            
            # Fix: Calculate thumb_angle before using it in debug data
            # Define angle calculation function if not already available in this scope
            def angle_between_points(a, b, c):
                a = np.array(a)
                b = np.array(b)
                c = np.array(c)
                ba = a - b
                bc = c - b
                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
                angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
                return angle
            
            # Calculate thumb angle
            thumb_angle = angle_between_points(thumb_mcp, thumb_ip, thumb_tip)
            
            if self.debug_mode:
                # אחסן את המידע בתוך self.detection_data במקום להדפיס
                self.detection_data['thumb_angle_debug'] = {
                    'angle': thumb_angle,
                    'finger_spacing': finger_spacings,
                    'all_extended': all_fingers_extended,
                    'evenly_spaced': fingers_evenly_spaced
                }
            
            stop_gesture_detected = all_fingers_extended and fingers_evenly_spaced
            self.detection_data['stop_sign_detected'] = stop_gesture_detected
            
            return stop_gesture_detected
        
        except Exception as e:
            print(f"Error in _detect_stop_sign_gesture: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _extract_controls_from_landmarks(self, landmarks, frame, controls):
        """
        Extract control values from hand landmarks with improved detection.
        """
        h, w, c = frame.shape
        landmark_points = []
        for lm in landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            landmark_points.append((x, y))
        
        # הגדרה של נקודות מפתח
        wrist = landmark_points[0]
        thumb_tip = landmark_points[4]
        thumb_ip = landmark_points[3]   # מפרק אמצעי של האגודל
        thumb_mcp = landmark_points[2]  # בסיס האגודל
        index_tip = landmark_points[8]
        index_mcp = landmark_points[5]
        middle_tip = landmark_points[12]
        middle_mcp = landmark_points[9]
        ring_tip = landmark_points[16]
        ring_mcp = landmark_points[13]
        pinky_tip = landmark_points[20]
        pinky_mcp = landmark_points[17]
        
        # Store hand position data
        self.detection_data['hand_position_y'] = wrist[1]
        
        # ==================== THROTTLE DETECTION ====================
        normalized_y = wrist[1] / h
        raw_throttle = 1.0 - normalized_y
        self.detection_data['normalized_y'] = normalized_y
        self.detection_data['raw_throttle'] = raw_throttle
        controls['throttle'] = self.prev_throttle * self.throttle_smoothing + raw_throttle * (1 - self.throttle_smoothing)
        self.prev_throttle = controls['throttle']
        
        # ==================== STEERING DETECTION ====================
        # Calculate the angle between thumb tip and wrist
        thumb_to_wrist_dx = thumb_tip[0] - wrist[0]
        thumb_to_wrist_dy = thumb_tip[1] - wrist[1]
        thumb_to_wrist_angle = np.degrees(np.arctan2(-thumb_to_wrist_dy, thumb_to_wrist_dx))
        if thumb_to_wrist_angle < 0:
            thumb_to_wrist_angle += 360
            
        # Store the thumb angle in detection data for display
        self.detection_data['thumb_angle'] = thumb_to_wrist_angle

        # Steering only in upper 180 degrees (thumb pointing up/right/left, not down)
        # Active range: 0° (right) to 180° (left), i.e., top half of the circle
        raw_steering = 0.0
        if 0 <= thumb_to_wrist_angle <= 180:
            # 0° (right) -> 1, 90° (up) -> 0, 180° (left) -> -1
            raw_steering = np.cos(np.deg2rad(thumb_to_wrist_angle))
            # Clamp for safety
            raw_steering = np.clip(raw_steering, -1.0, 1.0)
        # If thumb is pointing down (180° < angle < 360°), steering remains 0

        # Apply smoothing to the steering
        controls['steering'] = self.prev_steering * self.steering_smoothing + raw_steering * (1 - self.steering_smoothing)
        self.prev_steering = controls['steering']
        
        # בדיקה לאגרוף (מחווה לבלימה) - כל האצבעות מכופפות
        index_curled = index_tip[1] > index_mcp[1]  # האצבע מכופפת אם הקצה מתחת למפרק
        middle_curled = middle_tip[1] > middle_mcp[1]
        ring_curled = ring_tip[1] > ring_mcp[1]
        pinky_curled = pinky_tip[1] > pinky_mcp[1]

        # מרחקים לחישובי זיהוי מחוות
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

        # בדיקה אם האגודל זקוף בהקשר של מחוות "אגודל למעלה"
        thumb_pointing_up = self._check_thumb_extended(thumb_tip, thumb_ip, thumb_mcp, wrist, context="thumb_up")

        # בדיקה נוספת: האם האגודל קרוב לאצבע המורה (מאפיין אגרוף)
        thumb_close_to_index = thumb_tip_to_index_tip < 0.4 * thumb_tip_to_wrist  # מרחק קטן = אגודל קרוב לאצבע

        # עדכון ההגדרה של אגרוף - אגודל לא זקוף וקרוב לאצבע המורה + כל האצבעות מכופפות
        fist_detected = index_curled and middle_curled and ring_curled and pinky_curled and not thumb_pointing_up and thumb_close_to_index
        
        # הוספת מידע דיבאג
        if self.debug_mode:
            cv2.putText(frame, f"Fist: {fist_detected}", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, f"ThumbCloseToIndex: {thumb_close_to_index}", (10, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, f"ThumbUp: {thumb_pointing_up}", (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            cv2.putText(frame, f"Thumb>Fingers: {thumb_higher_than_fingers}", (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            cv2.putText(frame, f"Thumb-Index dist: {thumb_tip_to_index_tip:.1f}", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            cv2.putText(frame, f"Thumb angle: {thumb_angle:.1f}", (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            cv2.putText(frame, f"ThumbToWristAngle: {thumb_to_wrist_angle:.1f}", (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
            cv2.putText(frame, f"Steering: {controls['steering']:.2f}", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

        boost_gesture = thumb_pointing_up and index_curled and middle_curled and ring_curled and pinky_curled

        # בדיקת מחוות stop sign (יד פתוחה)
        stop_sign = self._detect_stop_sign_gesture(landmark_points, frame)

        # עדכון ערכי השליטה על בסיס החלטות היגוי ואיתור מחוות
        if fist_detected:
            controls['braking'] = True
            controls['gesture_name'] = 'Brake (Fist)'
        elif stop_sign:
            controls['braking'] = True
            controls['gesture_name'] = 'Stop (Open Palm)'
        elif boost_gesture:
            controls['boost'] = True
            controls['gesture_name'] = 'Boost (Thumb Up)'
        elif abs(controls['steering']) > 0.3:
            if controls['steering'] < -0.3:
                controls['gesture_name'] = f'Turning Left ({controls["steering"]:.2f})'
            else:
                controls['gesture_name'] = f'Turning Right ({controls["steering"]:.2f})'
        else:
            controls['gesture_name'] = 'Forward'
        
        return controls

    def _check_thumb_extended(self, thumb_tip, thumb_ip, thumb_mcp, wrist, context="thumb_up"):
        """
        בדיקה מאוחדת האם האגודל זקוף.
        מחזירה True אם האגודל זקוף, False אם הוא מקופל.
        
        Args:
            thumb_tip, thumb_ip, thumb_mcp, wrist: נקודות המפרקים של האגודל ושורש כף היד.
            context (str): סוג המחווה ("thumb_up" עבור בוסט, "open_palm" עבור עצור).
        """
        # חישוב מרחקים
        thumb_tip_to_ip = np.linalg.norm(np.array(thumb_tip) - np.array(thumb_ip))
        thumb_ip_to_mcp = np.linalg.norm(np.array(thumb_ip) - np.array(thumb_mcp))
        thumb_tip_to_wrist = np.linalg.norm(np.array(thumb_tip) - np.array(wrist))
        
        # חישוב יחס המרחקים
        distance_ratio = thumb_tip_to_ip / thumb_ip_to_mcp if thumb_ip_to_mcp > 0 else 0
        
        # חישוב זווית בין מפרקי האגודל (MCP -> IP -> TIP)
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
        
        # חישוב כיוון האגודל ביחס לשורש כף היד
        thumb_to_wrist_dx = thumb_tip[0] - wrist[0]
        thumb_to_wrist_dy = thumb_tip[1] - wrist[1]
        thumb_to_wrist_angle = np.degrees(np.arctan2(-thumb_to_wrist_dy, thumb_to_wrist_dx))
        if thumb_to_wrist_angle < 0:
            thumb_to_wrist_angle += 360
        
        # עבור "אגודל למעלה" (בוסט), האגודל חייב להצביע כלפי מעלה
        if context == "thumb_up":
            thumb_pointing_upward = (270 <= thumb_to_wrist_angle <= 360) or (0 <= thumb_to_wrist_angle <= 90)
            thumb_extended = (
                distance_ratio > 0.7 and
                thumb_angle > 120 and
                thumb_pointing_upward
            )
        # עבור "כף יד פתוחה" (עצור), האגודל צריך להיות פרוש הצידה מהכף
        else:  # context == "open_palm"
            # בדוק אם האגודל פרוש על ידי השוואת הזווית שלו לזווית הכף
            index_mcp = (self.image_width - wrist[0], wrist[1])  # הערכה של מפרק האצבע המורה
            wrist_to_index_dx = index_mcp[0] - wrist[0]
            wrist_to_index_dy = index_mcp[1] - wrist[1]
            wrist_to_index_angle = np.degrees(np.arctan2(wrist_to_index_dy, wrist_to_index_dx))
            if wrist_to_index_angle < 0:
                wrist_to_index_angle += 360
            
            # חישוב ההפרש בזווית בין האגודל לכיוון הכף
            angle_diff = abs(thumb_to_wrist_angle - wrist_to_index_angle)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            # האגודל נחשב "פרוש" אם הוא בזווית מספיק שונה מהכף (כלומר, לא מקופל פנימה)
            thumb_spread_out = angle_diff > 30  # ניתן לשנות את הסף הזה לפי הצורך
            thumb_extended = (
                distance_ratio > 0.7 and
                thumb_angle > 90 and  # שינוי מ-120 ל-90 כדי להתאים למחווה טבעית יותר
                thumb_spread_out
            )
        
        return thumb_extended

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