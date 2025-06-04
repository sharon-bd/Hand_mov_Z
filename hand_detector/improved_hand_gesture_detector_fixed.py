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
            # Handle invalid or empty frames
            if frame is None or frame.size == 0:
                default_controls = {
                    'steering': 0.0,
                    'throttle': 0.0,
                    'braking': False,
                    'boost': False,
                    'gesture_name': 'No frame data',
                    'speed': 0.0,
                    'direction': 0.0
                }
                empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                empty_panel = self._create_error_panel()
                return default_controls, empty_frame, empty_panel
            
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
            
            # Ensure controls is always a dictionary
            if not isinstance(controls, dict):
                controls = {
                    'steering': 0.0,
                    'throttle': 0.0,
                    'braking': False,
                    'boost': False,
                    'gesture_name': 'Error: Invalid control format',
                    'speed': 0.0,
                    'direction': 0.0
                }
            return controls, processed_frame, data_panel
        except Exception as e:
            print(f"Error in gesture detection: {e}")
            import traceback
            traceback.print_exc()
            
            # Return default values on error
            default_controls = {
                'steering': 0.0,
                'throttle': 0.0,
                'braking': False,
                'boost': False,
                'gesture_name': f'Error: {str(e)}',
                'speed': 0.0,
                'direction': 0.0
            }
            error_panel = self._create_error_panel()
            empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            return default_controls, empty_frame, error_panel

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
            
            # 1. בדיקה שכל האצבעות פרושות (לא מכופפות) - החמרה
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
                
                # החמרה: אצבע חייבת להיות פרושה מאוד
                is_extended = False
                if dist_mcp > 0:
                    is_extended = dist_tip > dist_mcp * 1.5  # הוגדל מ-1.2 ל-1.5
                
                # בדיקה נוספת: האצבע חייבת להיות רחוקה מכף היד
                distance_from_palm = dist_tip
                is_extended = is_extended and distance_from_palm > 0.2  # הוגדל מ-0.15 ל-0.2
                
                finger_extended.append(is_extended)
                self.detection_data['finger_status'][name] = is_extended
            
            # 2. בדיקת האגודל - הסרת כל התנאים המגבילים על הזווית
            thumb_ip = landmark_points[3]
            
            # בדיקה פשוטה לאגודל - רק שהוא פרוש, ללא קשר לזווית
            thumb_tip_to_ip = np.linalg.norm(np.array(thumb_tip) - np.array(thumb_ip))
            thumb_ip_to_mcp = np.linalg.norm(np.array(thumb_ip) - np.array(thumb_mcp))
            thumb_tip_to_wrist = np.linalg.norm(np.array(thumb_tip) - np.array(wrist))
            
            # יחס המרחקים - האגודל פרוש
            distance_ratio = thumb_tip_to_ip / thumb_ip_to_mcp if thumb_ip_to_mcp > 0 else 0
            
            # זווית בין מפרקי האגודל
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
            
            # אגודל פרוש - רק על בסיס מרחק וזווית, ללא הגבלת כיוון
            thumb_extended = (
                distance_ratio > 1.0 and     # האגודל פרוש מספיק
                thumb_angle > 120 and        # זווית פתוחה
                thumb_tip_to_wrist > 0.15    # רחוק מכף היד
            )
            
            self.detection_data['finger_status']["thumb"] = thumb_extended
            
            # 3. כל האצבעות חייבות להיות פרושות - ללא יצנים
            all_fingers_extended = all(finger_extended) and thumb_extended
            self.detection_data['all_fingers_extended'] = all_fingers_extended
            
            # 4. בדיקה שהאצבעות פרושות במרחק סביר - מחמיר יותר
            finger_tips = [index_tip, middle_tip, ring_tip, pinky_tip]
            finger_spacings = []
            
            for i in range(len(finger_tips) - 1):
                dx = float(finger_tips[i][0] - finger_tips[i+1][0])
                dy = float(finger_tips[i][1] - finger_tips[i+1][1])
                spacing = float(np.sqrt(float(dx)**2 + float(dy)**2))
                finger_spacings.append(spacing)
            
            # כל המרווחים חייבים להיות מעל מינימום
            fingers_evenly_spaced = True
            min_spacing_required = 0.08  # הוגדל מ-0.05 ל-0.08
            
            if len(finger_spacings) > 0:
                min_spacing = min(finger_spacings)
                fingers_evenly_spaced = min_spacing > min_spacing_required
            
            # החלטה סופית - כל התנאים חייבים להתקיים
            stop_gesture_detected = all_fingers_extended and fingers_evenly_spaced
            
            self.detection_data['stop_sign_detected'] = stop_gesture_detected
            
            if self.debug_mode and stop_gesture_detected:
                print(f"🛑 STOP detected: All fingers extended + proper spacing")
            
            return stop_gesture_detected
        
        except Exception as e:
            print(f"Error in _detect_stop_sign_gesture: {e}")
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

        # Enhanced steering control with dead zone and non-linear response
        raw_steering = 0.0
        if 0 <= thumb_to_wrist_angle <= 180:
            # Create a dead zone around vertical (90°) for more stable neutral position
            # Dead zone: 80° to 100°
            if 80 <= thumb_to_wrist_angle <= 100:
                raw_steering = 0.0
            else:
                # Non-linear steering response for better control
                # Map 0-80° to positive steering (right)
                # Map 100-180° to negative steering (left)
                if thumb_to_wrist_angle < 80:
                    # Exponential response for right turns
                    normalized = (80 - thumb_to_wrist_angle) / 80.0
                    raw_steering = normalized ** 1.5  # Exponential curve
                else:  # > 100°
                    # Exponential response for left turns
                    normalized = (thumb_to_wrist_angle - 100) / 80.0
                    raw_steering = -(normalized ** 1.5)  # Exponential curve
                
                # Scale the steering to full range
                raw_steering *= 1.0  # Maximum steering value

        # Add steering limits for safety
        raw_steering = np.clip(raw_steering, -1.0, 1.0)

        # Apply smoothing with variable smoothing factor
        # More smoothing at center, less at extremes
        center_smoothing = 0.8  # High smoothing near center
        edge_smoothing = 0.3    # Low smoothing at edges
        smoothing_factor = center_smoothing - (abs(raw_steering) * (center_smoothing - edge_smoothing))
        
        controls['steering'] = (self.prev_steering * smoothing_factor + 
                              raw_steering * (1 - smoothing_factor))
        self.prev_steering = controls['steering']

        # Enhanced debug visualization
        if self.debug_mode:
            # Draw steering zones
            center_x, center_y = wrist[0], wrist[1]
            radius = 50
            
            # Draw dead zone (yellow)
            cv2.ellipse(frame, (center_x, center_y), (radius, radius), 
                       0, -100, -80, (0, 255, 255), 2)
            
            # Draw active zones (green for right, red for left)
            cv2.ellipse(frame, (center_x, center_y), (radius, radius), 
                       0, -80, 0, (0, 255, 0), 2)  # Right zone
            cv2.ellipse(frame, (center_x, center_y), (radius, radius), 
                       0, -180, -100, (0, 0, 255), 2)  # Left zone
            
            # Draw current thumb angle
            angle_rad = np.radians(-thumb_to_wrist_angle)
            end_x = int(center_x + radius * np.cos(angle_rad))
            end_y = int(center_y + radius * np.sin(angle_rad))
            cv2.line(frame, (center_x, center_y), (end_x, end_y), (255, 255, 255), 2)
        
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
            
        # Calculate thumb angle
        thumb_angle = angle_between_points(thumb_mcp, thumb_ip, thumb_tip)
        
        if self.debug_mode:
            # אחסן את המידע בתוך self.detection_data במקום להדפיס
            self.detection_data['thumb_angle_debug'] = {
                'angle': thumb_angle,
                # Remove references to undefined variables
            }
        
        # Call the stop sign gesture detection function
        stop_gesture_detected = self._detect_stop_sign_gesture(landmark_points, frame)
        self.detection_data['stop_sign_detected'] = stop_gesture_detected
        
        return stop_gesture_detected
    
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

        # Enhanced steering control with dead zone and non-linear response
        raw_steering = 0.0
        if 0 <= thumb_to_wrist_angle <= 180:
            # Create a dead zone around vertical (90°) for more stable neutral position
            # Dead zone: 80° to 100°
            if 80 <= thumb_to_wrist_angle <= 100:
                raw_steering = 0.0
            else:
                # Non-linear steering response for better control
                # Map 0-80° to positive steering (right)
                # Map 100-180° to negative steering (left)
                if thumb_to_wrist_angle < 80:
                    # Exponential response for right turns
                    normalized = (80 - thumb_to_wrist_angle) / 80.0
                    raw_steering = normalized ** 1.5  # Exponential curve
                else:  # > 100°
                    # Exponential response for left turns
                    normalized = (thumb_to_wrist_angle - 100) / 80.0
                    raw_steering = -(normalized ** 1.5)  # Exponential curve
                
                # Scale the steering to full range
                raw_steering *= 1.0  # Maximum steering value

        # Add steering limits for safety
        raw_steering = np.clip(raw_steering, -1.0, 1.0)

        # Apply smoothing with variable smoothing factor
        # More smoothing at center, less at extremes
        center_smoothing = 0.8  # High smoothing near center
        edge_smoothing = 0.3    # Low smoothing at edges
        smoothing_factor = center_smoothing - (abs(raw_steering) * (center_smoothing - edge_smoothing))
        
        controls['steering'] = (self.prev_steering * smoothing_factor + 
                              raw_steering * (1 - smoothing_factor))
        self.prev_steering = controls['steering']

        # Enhanced debug visualization
        if self.debug_mode:
            # Draw steering zones
            center_x, center_y = wrist[0], wrist[1]
            radius = 50
            
            # Draw dead zone (yellow)
            cv2.ellipse(frame, (center_x, center_y), (radius, radius), 
                       0, -100, -80, (0, 255, 255), 2)
            
            # Draw active zones (green for right, red for left)
            cv2.ellipse(frame, (center_x, center_y), (radius, radius), 
                       0, -80, 0, (0, 255, 0), 2)  # Right zone
            cv2.ellipse(frame, (center_x, center_y), (radius, radius), 
                       0, -180, -100, (0, 0, 255), 2)  # Left zone
            
            # Draw current thumb angle
            angle_rad = np.radians(-thumb_to_wrist_angle)
            end_x = int(center_x + radius * np.cos(angle_rad))
            end_y = int(center_y + radius * np.sin(angle_rad))
            cv2.line(frame, (center_x, center_y), (end_x, end_y), (255, 255, 255), 2)
        
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
            
        # Calculate thumb angle
        thumb_angle = angle_between_points(thumb_mcp, thumb_ip, thumb_tip)
        
        # ==================== GESTURE DETECTION ====================
        # בדיקת מחוות stop sign (יד פתוחה) - רק כל האצבעות פרושות
        stop_sign = self._detect_stop_sign_gesture(landmark_points, frame)
        
        # בדיקת אגרוף לבלימה
        fist_detected = (index_curled and middle_curled and ring_curled and pinky_curled and
                        distance_ratio < 0.8)  # האגודל גם מכופף
        
        # בדיקת בוסט (אגודל למעלה)
        boost_gesture = (distance_ratio > 1.2 and thumb_angle > 140 and
                        270 <= thumb_to_wrist_angle <= 360 or 0 <= thumb_to_wrist_angle <= 90)
        
        # עדכון ערכי השליטה על בסיס החלטות היגוי ואיתור מחוות
        if fist_detected:
            controls['braking'] = True
            controls['gesture_name'] = 'Brake (Fist)'
        elif stop_sign:
            # ודא שכל האצבעות זקופות (כולל האגודל)
            if all([
                self.detection_data['finger_status'].get('thumb', False),
                self.detection_data['finger_status'].get('index', False),
                self.detection_data['finger_status'].get('middle', False),
                self.detection_data['finger_status'].get('ring', False),
                self.detection_data['finger_status'].get('pinky', False)
            ]):
                controls['braking'] = True
                controls['gesture_name'] = 'Stop (Open Palm)'
            else:
                # לא כל האצבעות זקופות - לא להפעיל Stop
                controls['gesture_name'] = 'Forward'
        elif boost_gesture:
            controls['boost'] = True
            controls['gesture_name'] = 'Boost (Thumb Up)'
        else:
            controls['gesture_name'] = 'Forward'
        
        # Store detection data
        self.detection_data['fist_detected'] = fist_detected
        self.detection_data['stop_sign_detected'] = stop_sign
        
        return controls