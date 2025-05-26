import cv2
import mediapipe as mp
import numpy as np
import math
import time

class ImprovedHandGestureDetector:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Track only one hand as per MD spec
            min_detection_confidence=0.9,  # High accuracy requirement from MD
            min_tracking_confidence=0.8
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Control smoothing parameters
        self.steering_history = []
        self.throttle_history = []
        self.history_size = 5
        
        # Gesture detection parameters
        self.gesture_threshold = 0.85  # 90% accuracy requirement from MD
        self.stability_frames = 3  # Filter unstable gestures
        
        # Current gesture state
        self.current_gesture = "unknown"
        self.gesture_confidence = 0.0
        self.gesture_stability_count = 0
        
        # Hand position calibration
        self.hand_center_x = 0.5  # Normalized center position
        self.hand_center_y = 0.5
        self.calibration_samples = []
        self.is_calibrated = False
        
        # FPS tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def detect_gestures(self, frame):
        """
        Main detection method that processes frame and returns controls
        Returns: (controls_dict, processed_frame, data_panel)
        """
        if frame is None:
            return self._get_default_controls(), frame, None
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # Initialize default controls
        controls = self._get_default_controls()
        
        # Process hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks as per MD requirement
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
                # Extract controls from hand landmarks
                controls = self._extract_controls_from_landmarks(hand_landmarks)
                
                # Update gesture recognition
                self._update_gesture_recognition(hand_landmarks)
                
                break  # Process only first hand
        else:
            # Handle absence of hand detection as per MD requirement
            self.current_gesture = "no_hand"
            self.gesture_confidence = 0.0
            controls = self._handle_no_hand_detected()
        
        # Create data panel for display
        data_panel = self._create_data_panel(controls)
        
        # Update FPS counter
        self._update_fps()
        
        return controls, frame, data_panel
    
    def _extract_controls_from_landmarks(self, landmarks):
        """
        Extract car controls from hand landmarks according to MD specification
        """
        # Convert landmarks to numpy array for easier processing
        landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        
        # Calculate hand center and rotation for steering
        steering = self._calculate_steering(landmark_array)
        
        # Calculate throttle based on hand height
        throttle = self._calculate_throttle(landmark_array)
        
        # Detect specific gestures
        is_braking = self._detect_fist_gesture(landmark_array)
        is_boost = self._detect_thumb_up_gesture(landmark_array)
        is_stop = self._detect_open_palm_gesture(landmark_array)
        
        # Apply gesture smoothing as per MD requirement
        steering = self._smooth_control(steering, self.steering_history)
        throttle = self._smooth_control(throttle, self.throttle_history)
        
        # Create controls dictionary
        controls = {
            'steering': steering,
            'throttle': throttle if not is_stop else 0.0,
            'braking': is_braking,
            'boost': is_boost,
            'emergency_stop': is_stop,
            'gesture_name': self.current_gesture,
            'confidence': self.gesture_confidence
        }
        
        return controls
    
    def _calculate_steering(self, landmarks):
        """
        Calculate steering based on hand tilt (rotation) as per MD spec
        """
        # Use wrist (0) and middle finger MCP (9) to calculate hand rotation
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        
        # Calculate angle of hand tilt
        dx = middle_mcp[0] - wrist[0]
        dy = middle_mcp[1] - wrist[1]
        
        # Calculate rotation angle in radians
        rotation = math.atan2(dx, dy)
        
        # Convert to steering value (-1.0 to 1.0)
        # Normalize rotation from [-π/4, π/4] to [-1, 1]
        max_rotation = math.pi / 4  # 45 degrees max tilt
        steering = np.clip(rotation / max_rotation, -1.0, 1.0)
        
        return steering
    
    def _calculate_throttle(self, landmarks):
        """
        Calculate throttle based on hand height as per MD spec
        """
        # Use wrist position for hand height
        wrist_y = landmarks[0][1]
        
        # Normalize hand height to throttle (0.0 to 1.0)
        # Lower position = higher throttle
        # Assuming normalized coordinates where 0 is top, 1 is bottom
        throttle = 1.0 - wrist_y
        throttle = np.clip(throttle, 0.0, 1.0)
        
        return throttle
    
    def _detect_fist_gesture(self, landmarks):
        """
        Detect fist gesture for braking as per MD spec
        """
        # Check if all fingers are curled (fist)
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
        finger_pips = [3, 6, 10, 14, 18]  # Corresponding PIP joints
        
        curled_fingers = 0
        
        for tip, pip in zip(finger_tips, finger_pips):
            # Check if fingertip is below PIP joint (finger curled)
            if landmarks[tip][1] > landmarks[pip][1]:
                curled_fingers += 1
        
        # Fist detected if 4 or more fingers are curled
        is_fist = curled_fingers >= 4
        
        if is_fist:
            self.current_gesture = "fist_brake"
            self.gesture_confidence = curled_fingers / 5.0
        
        return is_fist
    
    def _detect_thumb_up_gesture(self, landmarks):
        """
        Detect thumb up gesture for boost as per MD spec
        """
        # Check thumb position (tip should be above other finger tips)
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        
        # Check if thumb is extended upward
        thumb_extended = thumb_tip[1] < thumb_mcp[1]
        
        # Check if other fingers are curled
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
        finger_pips = [6, 10, 14, 18]  # Corresponding PIP joints
        
        curled_fingers = 0
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip][1] > landmarks[pip][1]:
                curled_fingers += 1
        
        # Thumb up if thumb extended and other fingers curled
        is_thumb_up = thumb_extended and curled_fingers >= 3
        
        if is_thumb_up:
            self.current_gesture = "thumb_up_boost"
            self.gesture_confidence = 0.9 if curled_fingers == 4 else 0.7
        
        return is_thumb_up
    
    def _detect_open_palm_gesture(self, landmarks):
        """
        Detect open palm gesture for emergency stop as per MD spec
        """
        # Check if all fingers are extended
        finger_tips = [4, 8, 12, 16, 20]  # All fingertips
        finger_pips = [3, 6, 10, 14, 18]  # Corresponding PIP joints
        
        extended_fingers = 0
        
        for tip, pip in zip(finger_tips, finger_pips):
            # Check if fingertip is above PIP joint (finger extended)
            if landmarks[tip][1] < landmarks[pip][1]:
                extended_fingers += 1
        
        # Open palm if all or most fingers are extended
        is_open_palm = extended_fingers >= 4
        
        if is_open_palm:
            self.current_gesture = "open_palm_stop"
            self.gesture_confidence = extended_fingers / 5.0
        
        return is_open_palm
    
    def _smooth_control(self, value, history):
        """
        Apply smoothing to control inputs to prevent jerky movement as per MD requirement
        """
        history.append(value)
        if len(history) > self.history_size:
            history.pop(0)
        
        # Return weighted average with more weight on recent values
        weights = np.array([i + 1 for i in range(len(history))])
        weights = weights / np.sum(weights)
        
        return np.average(history, weights=weights)
    
    def _update_gesture_recognition(self, landmarks):
        """
        Update gesture recognition with stability filtering as per MD requirement
        """
        # This method updates the current gesture and applies stability filtering
        # Gesture confidence is already set in individual detection methods
        
        # Apply stability threshold
        if self.gesture_confidence >= self.gesture_threshold:
            self.gesture_stability_count += 1
        else:
            self.gesture_stability_count = 0
        
        # Only update gesture if it's stable for required frames
        if self.gesture_stability_count < self.stability_frames:
            # Keep previous gesture if new one isn't stable enough
            if not hasattr(self, 'stable_gesture'):
                self.stable_gesture = "unknown"
        else:
            self.stable_gesture = self.current_gesture
    
    def _handle_no_hand_detected(self):
        """
        Handle absence of hand detection by gradually slowing the car as per MD requirement
        """
        # Gradually reduce throttle when no hand is detected
        controls = self._get_default_controls()
        controls['throttle'] = 0.0  # No throttle
        controls['gradual_stop'] = True  # Flag for gradual stopping
        controls['gesture_name'] = "no_hand_detected"
        
        return controls
    
    def _get_default_controls(self):
        """
        Return default control values
        """
        return {
            'steering': 0.0,
            'throttle': 0.0,
            'braking': False,
            'boost': False,
            'emergency_stop': False,
            'gesture_name': 'unknown',
            'confidence': 0.0,
            'gradual_stop': False
        }
    
    def _create_data_panel(self, controls):
        """
        Create data panel for display with gesture information
        """
        data_panel = np.zeros((300, 400, 3), dtype=np.uint8)
        
        # Display current gesture and confidence
        cv2.putText(data_panel, f"Gesture: {controls['gesture_name']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(data_panel, f"Confidence: {controls['confidence']:.2f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display control values
        cv2.putText(data_panel, f"Steering: {controls['steering']:.2f}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(data_panel, f"Throttle: {controls['throttle']:.2f}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(data_panel, f"Braking: {controls['braking']}", 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(data_panel, f"Boost: {controls['boost']}", 
                   (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(data_panel, f"Emergency Stop: {controls['emergency_stop']}", 
                   (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Display FPS
        cv2.putText(data_panel, f"FPS: {self.current_fps:.1f}", 
                   (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return data_panel
    
    def _update_fps(self):
        """
        Update FPS counter to maintain minimum 15 FPS as per MD requirement
        """
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
            
            # Check if FPS is below minimum requirement
            if self.current_fps < 15:
                print(f"Warning: FPS ({self.current_fps:.1f}) below minimum requirement (15 FPS)")
    
    def calibrate_hand_position(self, landmarks):
        """
        Calibrate hand center position for better control mapping
        """
        # Calculate hand center
        wrist = landmarks[0]
        self.calibration_samples.append([wrist[0], wrist[1]])
        
        if len(self.calibration_samples) >= 30:  # 30 samples for calibration
            # Calculate average center position
            samples_array = np.array(self.calibration_samples)
            self.hand_center_x = np.mean(samples_array[:, 0])
            self.hand_center_y = np.mean(samples_array[:, 1])
            self.is_calibrated = True
            print(f"Hand position calibrated: center at ({self.hand_center_x:.2f}, {self.hand_center_y:.2f})")
    
    def get_performance_metrics(self):
        """
        Get performance metrics as per MD analytics requirement
        """
        return {
            'current_fps': self.current_fps,
            'gesture_confidence': self.gesture_confidence,
            'is_hand_detected': self.current_gesture != "no_hand",
            'gesture_stability': self.gesture_stability_count >= self.stability_frames,
            'current_gesture': self.current_gesture
        }
