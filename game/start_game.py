import os
import time
import logging
import pygame
import random
import cv2
import sys
import math

# Try to import the moving road module
try:
    from .moving_road import MovingRoadGenerator
    MOVING_ROAD_AVAILABLE = True
    print("✅ MovingRoad module imported successfully")
except ImportError:
    MOVING_ROAD_AVAILABLE = False
    print("⚠️ MovingRoad module not found - using built-in road animation")

# Try to import MediaPipe for real hand detection
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe imported successfully")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️ MediaPipe not found - hand detection will be disabled")

# Configuration constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60
CAPTION = "Hand Gesture Car Control"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Game modes configuration
GAME_MODES = {
    "normal": {
        "time_limit": 180,
        "obstacle_frequency": 0.02,
        "obstacle_speed": 200,
        "score_multiplier": 1.0
    },
    "easy": {
        "time_limit": 240,
        "obstacle_frequency": 0.01,
        "obstacle_speed": 150,
        "score_multiplier": 0.8
    },
    "hard": {
        "time_limit": 120,
        "obstacle_frequency": 0.03,
        "obstacle_speed": 250,
        "score_multiplier": 1.5
    }
}

OBSTACLE_FREQUENCY = 0.02
OBSTACLE_SPEED = 200
SCORE_PER_OBSTACLE = 10

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
PRIMARY = (59, 130, 246)
SECONDARY = (99, 102, 241)
ACCENT = (245, 158, 11)
SUCCESS = (34, 197, 94)
ERROR = (239, 68, 68)

logger = logging.getLogger(__name__)

# Simple stub classes to replace missing imports
class HandDetector:
    """Real hand detection using MediaPipe"""
    def __init__(self):
        if MEDIAPIPE_AVAILABLE:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.mp_draw = mp.solutions.drawing_utils
            self.enabled = True
            print("✅ Real HandDetector initialized with MediaPipe")
        else:
            self.enabled = False
            print("⚠️ HandDetector disabled - MediaPipe not available")
    
    def find_hands(self, frame):
        """Find hands in the frame"""
        if not self.enabled or frame is None:
            return frame, None
        
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.hands.process(rgb_frame)
            
            # Draw hand landmarks if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            return frame, results
            
        except Exception as e:
            print(f"⚠️ Error in hand detection: {e}")
            return frame, None
    
    def find_positions(self, frame, results):
        """Extract hand landmark positions"""
        if not self.enabled or not results or not results.multi_hand_landmarks:
            return None
        
        try:
            # Get the first hand detected
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract landmark positions
            landmarks = []
            h, w, c = frame.shape
            
            for id, lm in enumerate(hand_landmarks.landmark):
                # Convert normalized coordinates to pixel coordinates
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append([cx, cy])
            
            # Convert to normalized coordinates (0-1) for gesture recognition
            normalized_landmarks = []
            for landmark in landmarks:
                normalized_landmarks.append([landmark[0] / w, landmark[1] / h])
            
            return normalized_landmarks
            
        except Exception as e:
            print(f"⚠️ Error extracting landmark positions: {e}")
            return None
    
    def release(self):
        """Release resources"""
        if self.enabled and hasattr(self, 'hands'):
            try:
                self.hands.close()
                print("✅ HandDetector resources released")
            except:
                pass

class GestureRecognizer:
    """Enhanced gesture recognition for hand control with improved thumbs-up and fist detection"""
    def __init__(self):
        self.gesture_history = []
        self.max_history = 10  # For smoothing
        self.stability_threshold = 0.1  # For filtering unstable gestures
        self.detection_fps = 0
        self.last_detection_time = time.time()
        self.frame_count = 0
        
        # Gesture detection parameters
        self.steering_sensitivity = 1.5
        self.throttle_sensitivity = 1.2
        self.gesture_confidence_threshold = 0.7
        
        # Add current throttle state for continuous control
        self.current_throttle = 0.5  # Start at medium throttle
        self.last_throttle_time = time.time()
        
        # Throttle change rates
        self.acceleration_rate = 0.8  # Units per second when accelerating
        self.deceleration_rate = 1.0  # Units per second when decelerating
        
        print("✅ Enhanced GestureRecognizer initialized")
    
    def recognize_gestures(self, landmarks, frame_height):
        """
        Enhanced gesture recognition with smoothing and stability filtering
        
        Args:
            landmarks: Hand landmarks from MediaPipe
            frame_height: Height of the frame for normalization
            
        Returns:
            Dictionary with gesture commands and confidence scores
        """
        current_time = time.time()
        self.frame_count += 1
        
        # Calculate detection FPS
        if current_time - self.last_detection_time >= 1.0:
            self.detection_fps = self.frame_count
            self.frame_count = 0
            self.last_detection_time = current_time
        
        if not landmarks or len(landmarks) < 21:
            return self._get_default_gestures()
        
        try:
            # Extract key landmarks
            thumb_tip = landmarks[4]      # Thumb tip
            thumb_ip = landmarks[3]       # Thumb IP joint
            thumb_mcp = landmarks[2]      # Thumb MCP joint
            thumb_base = landmarks[1]     # Thumb base
            index_tip = landmarks[8]      # Index finger tip
            index_mcp = landmarks[5]      # Index finger MCP joint
            middle_tip = landmarks[12]    # Middle finger tip
            middle_mcp = landmarks[9]     # Middle finger MCP joint
            ring_tip = landmarks[16]      # Ring finger tip
            ring_mcp = landmarks[13]      # Ring finger MCP joint
            pinky_tip = landmarks[20]     # Pinky tip
            pinky_mcp = landmarks[17]     # Pinky MCP joint
            wrist = landmarks[0]          # Wrist
            
            # FIXED: Calculate hand center properly using wrist and middle finger MCP
            hand_center_x = (wrist[0] + middle_mcp[0]) / 2
            hand_center_y = (wrist[1] + middle_mcp[1]) / 2
            
            # Additional debug for hand center calculation
            if not hasattr(self, '_last_center_log') or current_time - self._last_center_log > 2.0:
                print(f"🎯 Hand center calculation: wrist_y={wrist[1]:.3f}, middle_mcp_y={middle_mcp[1]:.3f}, center_y={hand_center_y:.3f}")
                self._last_center_log = current_time
            
            # 1. STEERING - Hand tilt for left/right
            steering = self._calculate_steering(landmarks, wrist, index_mcp)
            
            # 2. THROTTLE - Hand height for speed (use calculated hand center)
            throttle = self._calculate_throttle(landmarks, frame_height, hand_center_y)
            
            # 3. BRAKE - Fist gesture detection
            braking = self._detect_fist(landmarks, thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip,
                                       thumb_ip, index_mcp, middle_mcp, ring_mcp, pinky_mcp)
            
            # 4. BOOST - Thumb up with other fingers curled
            boost = self._detect_thumbs_up(landmarks, thumb_tip, thumb_ip, index_tip, middle_tip, 
                                          ring_tip, pinky_tip, wrist)
            
            # 5. STOP - Open palm detection
            stop = self._detect_open_palm(landmarks, thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip,
                                         thumb_ip, index_mcp, middle_mcp, ring_mcp, pinky_mcp)
            
            # Create raw gesture data
            raw_gesture = {
                'steering': steering,
                'throttle': throttle,
                'braking': braking,
                'boost': boost,
                'stop': stop,
                'confidence': self._calculate_confidence(landmarks),
                'detection_fps': self.detection_fps,
                'landmarks': landmarks  # Store for history
            }
            
            # Apply gesture smoothing and stability filtering
            smoothed_gesture = self._apply_smoothing(raw_gesture)
            
            # Determine primary gesture name
            gesture_name = self._determine_gesture_name(smoothed_gesture)
            smoothed_gesture['gesture_name'] = gesture_name
            
            return smoothed_gesture
            
        except Exception as e:
            print(f"⚠️ Error in gesture recognition: {e}")
            return self._get_default_gestures()
    
    def _calculate_steering(self, landmarks, wrist, index_mcp):
        """Calculate steering based on thumb orientation - FIXED: Perfect 180° symmetry"""
        try:
            # Use thumb for steering calculation with perfect symmetry
            thumb_tip = landmarks[4]      # Thumb tip
            thumb_mcp = landmarks[2]      # Thumb MCP joint
            
            # Calculate thumb vector
            dx = thumb_tip[0] - thumb_mcp[0]  # Horizontal difference
            dy = thumb_tip[1] - thumb_mcp[1]  # Vertical difference (Y increases downward)
            
            # Calculate angle in degrees relative to vertical up
            # 0° = thumb straight up (vertical)
            # +90° = thumb fully right
            # -90° = thumb fully left
            # ±180° = thumb straight down
            angle = math.degrees(math.atan2(dx, -dy))  # Negative dy for 0° up
            
            # Create perfect symmetry in the upper 180 degrees
            # Work only with the range -90° to +90° (upper half)
            if angle > 90:
                angle = 180 - angle    # Map 90°-180° to 90°-0°
            elif angle < -90:
                angle = -180 - angle   # Map -90°-(-180°) to 90°-0°
            
            # Now we have a range of -90° to +90° with perfect symmetry
            # 0° = straight (thumb up)
            # +90° = full right
            # -90° = full left
            
            # Convert to steering value in range -1.0 to +1.0
            max_angle = 75  # Maximum angle for full turn (less than 90 for comfort)
            steering = angle / max_angle
            
            # Clamp range to ±1.0
            steering = max(-1.0, min(1.0, steering))
            
            # Extended dead zone for more stable center
            dead_zone = 0.15  # 15% dead zone
            if abs(steering) < dead_zone:
                steering = 0.0
            else:
                # Smooth adjustment beyond dead zone
                sign = 1 if steering > 0 else -1
                adjusted_value = (abs(steering) - dead_zone) / (1.0 - dead_zone)
                steering = sign * adjusted_value
            
            # Apply sensitivity
            steering *= self.steering_sensitivity
            steering = max(-1.0, min(1.0, steering))  # Re-clamp
            
            return steering
            
        except Exception as e:
            print(f"⚠️ Error calculating steering: {e}")
            return 0.0
    
    def _calculate_throttle(self, landmarks, frame_height, hand_center_y):
        """
        Calculate throttle based on hand height with continuous acceleration/deceleration
        
        When hand is in upper half: Continuously increase speed
        When hand is in lower half: Decrease speed until 0.2
        """
        try:
            # Get current time for delta calculation
            current_time = time.time()
            dt = min(0.1, current_time - self.last_throttle_time)  # Cap at 100ms to avoid huge jumps
            self.last_throttle_time = current_time
            
            # Use the passed hand_center_y parameter properly
            # Normalize hand height (0 = top of frame, 1 = bottom of frame)
            normalized_height = hand_center_y
            
            # Debug output to see actual hand position
            if not hasattr(self, '_last_height_log') or current_time - self._last_height_log > 1.0:
                print(f"🎯 Hand height debug: center_y={hand_center_y:.3f}, normalized={normalized_height:.3f}")
                self._last_height_log = current_time
            
            # Determine if hand is in upper or lower half of the frame
            if normalized_height < 0.5:  # Upper half - Accelerate
                # Calculate acceleration factor (faster acceleration when hand is higher)
                accel_factor = 1.0 - normalized_height * 2  # 1.0 at top, 0.0 at middle
                
                # Apply acceleration based on hand height
                throttle_change = self.acceleration_rate * dt * (0.5 + accel_factor)
                self.current_throttle += throttle_change
                
                # Debug output for significant changes
                if throttle_change > 0.05:
                    print(f"👆 Accelerating: +{throttle_change:.2f} (hand height: {normalized_height:.2f})")
                
            else:  # Lower half - Decelerate
                # Calculate deceleration factor (faster deceleration when hand is lower)
                decel_factor = (normalized_height - 0.5) * 2  # 0.0 at middle, 1.0 at bottom
                
                # Apply deceleration based on hand height
                throttle_change = self.deceleration_rate * dt * (0.5 + decel_factor)
                self.current_throttle -= throttle_change
                
                # Debug output for significant changes
                if throttle_change > 0.05:
                    print(f"👇 Decelerating: -{throttle_change:.2f} (hand height: {normalized_height:.2f})")
            
            # Clamp throttle value with minimum 0.2 and maximum 1.0
            self.current_throttle = max(0.2, min(1.0, self.current_throttle))
            
            return self.current_throttle
            
        except Exception as e:
            print(f"⚠️ Error calculating throttle: {e}")
            return 0.5  # Default medium throttle
    
    def _detect_fist(self, landmarks, thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip,
                     thumb_ip, index_mcp, middle_mcp, ring_mcp, pinky_mcp):
        """Detect fist gesture for braking with stricter thumb and finger criteria"""
        try:
            thumb_mcp = landmarks[2]   # Thumb MCP joint
            thumb_base = landmarks[1]  # Thumb base
            
            # Stricter finger curl threshold
            finger_curl_threshold = 0.07  # Reduced for tighter curls
            finger_distances = [
                self._distance(index_tip, index_mcp),
                self._distance(middle_tip, middle_mcp),
                self._distance(ring_tip, ring_mcp),
                self._distance(pinky_tip, pinky_mcp)
            ]
            
            fingers_curled = all(dist < finger_curl_threshold for dist in finger_distances)
            
            # Enhanced thumb checks
            thumb_to_palm_distance = self._distance(thumb_tip, thumb_base)
            thumb_to_mcp_distance = self._distance(thumb_tip, thumb_mcp)
            
            # Ensure thumb is not raised above fingers
            thumb_not_raised = thumb_tip[1] >= min(index_tip[1], middle_tip[1], ring_tip[1], pinky_tip[1]) - 0.01
            
            # Thumb must be very close to palm
            thumb_close_to_palm = thumb_to_palm_distance < 0.10  # Reduced threshold
            
            # Thumb must not be extended sideways
            hand_center_x = (landmarks[0][0] + landmarks[9][0]) / 2
            thumb_horizontal_distance = abs(thumb_tip[0] - hand_center_x)
            thumb_not_extended_sideways = thumb_horizontal_distance < 0.12  # Reduced threshold
            
            # Ensure thumb is not aligned vertically (to avoid thumbs-up confusion)
            thumb_vector_x = thumb_tip[0] - thumb_mcp[0]
            thumb_vector_y = thumb_tip[1] - thumb_mcp[1]
            thumb_angle = abs(math.degrees(math.atan2(thumb_vector_x, -thumb_vector_y)))
            thumb_not_vertical = thumb_angle > 30  # Thumb must not be near vertical
            
            thumb_criteria = (
                thumb_close_to_palm and 
                thumb_not_raised and 
                thumb_not_extended_sideways and
                thumb_to_mcp_distance < 0.09 and  # Reduced threshold
                thumb_not_vertical
            )
            
            fist_detected = fingers_curled and thumb_criteria
            
            if fist_detected:
                current_time = time.time()
                if not hasattr(self, '_last_fist_log') or current_time - self._last_fist_log > 2.0:
                    print(f"✊ FIST detected! Fingers curled: {fingers_curled}, Thumb criteria: {thumb_criteria}")
                    print(f"   Finger distances: {[f'{d:.3f}' for d in finger_distances]}")
                    print(f"   Thumb to palm: {thumb_to_palm_distance:.3f}, Thumb angle: {thumb_angle:.1f}°")
                    self._last_fist_log = current_time
            
            return fist_detected
            
        except Exception as e:
            print(f"⚠️ Error in fist detection: {e}")
            return False
    
    def _detect_thumbs_up(self, landmarks, thumb_tip, thumb_ip, index_tip, middle_tip, 
                          ring_tip, pinky_tip, wrist):
        """Detect thumbs-up gesture for boost with stricter conditions"""
        try:
            thumb_mcp = landmarks[2]   # Thumb MCP joint
            thumb_base = landmarks[1]  # Thumb base
            
            # Thumb must be fully extended upward
            thumb_extended_up = (thumb_tip[1] < thumb_ip[1] < thumb_mcp[1] < thumb_base[1])
            
            # Thumb must be significantly higher than all other fingers
            min_finger_y = min(index_tip[1], middle_tip[1], ring_tip[1], pinky_tip[1])
            thumb_significantly_higher = thumb_tip[1] < min_finger_y - 0.10  # Increased threshold
            
            # Minimum distance between thumb tip and finger tips
            thumb_to_finger_distances = [
                self._distance(thumb_tip, index_tip),
                self._distance(thumb_tip, middle_tip),
                self._distance(thumb_tip, ring_tip),
                self._distance(thumb_tip, pinky_tip)
            ]
            min_thumb_to_finger_distance = min(thumb_to_finger_distances)
            thumb_separated = min_thumb_to_finger_distance > 0.12  # Ensure clear separation
            
            # Stricter finger curl checks
            index_curled = index_tip[1] > landmarks[6][1] + 0.03  # Increased threshold
            middle_curled = middle_tip[1] > landmarks[10][1] + 0.03
            ring_curled = ring_tip[1] > landmarks[14][1] + 0.03
            pinky_curled = pinky_tip[1] > landmarks[18][1] + 0.03
            
            # Thumb must be nearly vertical
            thumb_vector_x = thumb_tip[0] - thumb_mcp[0]
            thumb_vector_y = thumb_tip[1] - thumb_mcp[1]
            thumb_angle = abs(math.degrees(math.atan2(thumb_vector_x, -thumb_vector_y)))
            thumb_vertical = thumb_angle < 20  # Thumb must be near vertical
            
            # Thumb not too far sideways
            hand_center_x = (wrist[0] + landmarks[9][0]) / 2
            thumb_not_too_sideways = abs(thumb_tip[0] - hand_center_x) < 0.10  # Reduced threshold
            
            all_fingers_curled = index_curled and middle_curled and ring_curled and pinky_curled
            
            thumbs_up = (
                thumb_extended_up and 
                thumb_significantly_higher and 
                all_fingers_curled and 
                thumb_not_too_sideways and 
                thumb_separated and 
                thumb_vertical
            )
            
            if thumbs_up:
                current_time = time.time()
                if not hasattr(self, '_last_thumbs_up_log') or current_time - self._last_thumbs_up_log > 2.0:
                    print(f"👍 THUMBS UP detected! Boost activated!")
                    print(f"   Thumb higher: {thumb_significantly_higher}, Fingers curled: {all_fingers_curled}")
                    print(f"   Thumb to finger distance: {min_thumb_to_finger_distance:.3f}, Thumb angle: {thumb_angle:.1f}°")
                    self._last_thumbs_up_log = current_time
            
            return thumbs_up
            
        except Exception as e:
            print(f"⚠️ Error in thumbs up detection: {e}")
            return False
    
    def _detect_open_palm(self, landmarks, thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip,
                          thumb_ip, index_mcp, middle_mcp, ring_mcp, pinky_mcp):
        """Detect open palm gesture for stop"""
        try:
            # All fingers should be extended
            thumb_extended = self._distance(thumb_tip, thumb_ip) > 0.04
            index_extended = index_tip[1] < index_mcp[1] - 0.03
            middle_extended = middle_tip[1] < middle_mcp[1] - 0.03
            ring_extended = ring_tip[1] < ring_mcp[1] - 0.03
            pinky_extended = pinky_tip[1] < pinky_mcp[1] - 0.03
            
            # Fingers should be spread apart
            finger_spread = (
                self._distance(index_tip, middle_tip) > 0.03 and
                self._distance(middle_tip, ring_tip) > 0.03 and
                self._distance(ring_tip, pinky_tip) > 0.03
            )
            
            return (thumb_extended and index_extended and middle_extended and 
                   ring_extended and pinky_extended and finger_spread)
            
        except Exception:
            return False
    
    def _distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _calculate_confidence(self, landmarks):
        """
        Calculate confidence score based on landmark stability
        
        Args:
            landmarks: List of hand landmarks from MediaPipe
            
        Returns:
            float: Confidence score between 0 and 1
        """
        try:
            if len(self.gesture_history) < 3:
                return 0.5
            
            # Compare current landmarks with recent history
            recent_landmarks = [h.get('landmarks') for h in self.gesture_history[-3:] if h.get('landmarks')]
            if not recent_landmarks:
                return 0.5
                
            # Calculate stability as inverse of movement variance
            movement_variance = 0
            for i, landmark in enumerate(landmarks[:5]):  # Check first 5 landmarks
                if i < len(recent_landmarks[0]):
                    recent_positions = [rl[i] for rl in recent_landmarks if i < len(rl)]
                    if recent_positions:
                        x_var = sum((pos[0] - landmark[0])**2 for pos in recent_positions) / len(recent_positions)
                        y_var = sum((pos[1] - landmark[1])**2 for pos in recent_positions) / len(recent_positions)
                        movement_variance += x_var + y_var
            
            # Convert variance to confidence (lower variance = higher confidence)
            confidence = max(0.1, min(1.0, 1.0 - movement_variance * 10))
            return confidence
            
        except Exception:
            return 0.5
    
    def _apply_smoothing(self, raw_gesture):
        """Apply smoothing to reduce gesture jitter"""
        # Add current gesture to history
        gesture_with_landmarks = raw_gesture.copy()
        self.gesture_history.append(gesture_with_landmarks)
        
        # Limit history size
        if len(self.gesture_history) > self.max_history:
            self.gesture_history.pop(0)
        
        # Apply smoothing if we have enough history
        if len(self.gesture_history) < 3:
            return raw_gesture
        
        smoothed = {}
        
        # Smooth numerical values
        for key in ['steering', 'throttle']:
            if key in raw_gesture:
                recent_values = [h[key] for h in self.gesture_history[-5:] if key in h]
                if recent_values:
                    # Weighted average with more weight on recent values
                    weights = [0.1, 0.15, 0.2, 0.25, 0.3][-len(recent_values):]
                    smoothed[key] = sum(v * w for v, w in zip(recent_values, weights)) / sum(weights)
                else:
                    smoothed[key] = raw_gesture[key]
            else:
                smoothed[key] = raw_gesture.get(key, 0.0)
        
        # Smooth boolean values
        for key in ['braking', 'boost', 'stop']:
            if key in raw_gesture:
                recent_values = [h[key] for h in self.gesture_history[-3:] if key in h]
                if recent_values:
                    # Majority vote
                    smoothed[key] = sum(recent_values) > len(recent_values) / 2
                else:
                    smoothed[key] = raw_gesture[key]
            else:
                smoothed[key] = raw_gesture.get(key, False)
        
        # Apply stability filtering
        if len(self.gesture_history) >= 3:
            # Check if gesture is stable enough
            recent_steering = [h.get('steering', 0) for h in self.gesture_history[-3:]]
            steering_stability = max(recent_steering) - min(recent_steering)
            if steering_stability > self.stability_threshold:
                # Gesture is unstable, reduce sensitivity
                smoothed['steering'] *= 0.7
        
        # Copy other values
        for key in ['confidence', 'detection_fps', 'landmarks']:
            smoothed[key] = raw_gesture.get(key, 0)
        
        return smoothed
    
    def _determine_gesture_name(self, gesture):
        """Determine the primary gesture name"""
        if gesture.get('stop', False):
            return 'Stop (Open Palm)'
        elif gesture.get('braking', False):
            return 'Brake (Fist)'
        elif gesture.get('boost', False):
            return 'Boost (Thumbs Up)'
        elif abs(gesture.get('steering', 0)) > 0.3:
            direction = 'Right' if gesture['steering'] > 0 else 'Left'
            return f'Steering {direction}'
        elif gesture.get('throttle', 0) > 0.7:
            return 'High Throttle'
        elif gesture.get('throttle', 0) < 0.3:
            return 'Low Throttle'
        else:
            return 'Medium Throttle'  # Changed from 'Neutral' for consistency
    
    def get_detection_stats(self):
        """Get detection statistics"""
        return {
            'fps': self.detection_fps,
            'history_size': len(self.gesture_history),
            'stability_threshold': self.stability_threshold,
            'smoothing_enabled': len(self.gesture_history) >= 3
        }

class SoundManager:
    """Stub class for audio management"""
    def __init__(self):
        self.muted = False
    
    def toggle_mute(self):
        """
        Toggle the mute state
        
        Returns:
            bool: Current mute state
        """
        self._muted = not self.muted
        return self._muted
    
    def play(self, sound_name):
        """Play a specified sound"""
        pass
    
    def create_engine(self):
        """Create an engine sound"""
        pass
    
    def cleanup(self):
        """Clean up sound resources"""
        pass

class Button:
    """Simple button class"""
    def __init__(self, x, y, width, height, text, color, text_color, callback):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.text_color = text_color
        self.callback = callback
        self.font = pygame.font.Font(None, 24)
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                if self.callback:
                    self.callback()
    
    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
        pygame.draw.rect(screen, WHITE, self.rect, 2)
        
        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

def draw_text(screen, text, color, font, x, y, align="left"):
    """
    Draw text on the specified screen
    
    Args:
        screen: pygame.Surface, surface to draw on
        text: str, text to render
        color: tuple, text color
        font: pygame.font.Font, font to use
        x: int, x-coordinate position
        y: int, y-coordinate position
        align: str, text alignment ('left', 'center', 'right')
    """
    text_surface = font.render(text, True, color)
    
    if align == "center":
        text_rect = text_surface.get_rect(center=(x, y))
        screen.blit(text_surface, text_rect)
    elif align == "right":
        text_rect = text_surface.get_rect()
        text_rect.right = x
        text_rect.y = y
        screen.blit(text_surface, text_rect)
    else:
        screen.blit(text_surface, (x, y))

class Car:
    """Car class with proper physics and rendering"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 40
        self.height = 80
        self.speed = 0.0
        self.steering = 0.0
        self.throttle = 0.5
        self.braking = False
        self.boosting = False
        self.rotation = 0.0
        self.max_speed = 200
        self.health = 100
    
    def update(self, controls, dt, road_bounds=None):
        """Update car state"""
        if controls:
            new_steering = controls.get('steering', 0.0)
            self.throttle = controls.get('throttle', 0.5)
            self.braking = controls.get('braking', False)
            self.boosting = controls.get('boost', False)
            
            self.steering = new_steering
        
        # Simple physics
        if self.braking:
            self.speed = max(0.0, self.speed - 2.0 * dt)
        else:
            target_speed = self.throttle
            if self.boosting:
                target_speed = min(1.0, target_speed * 1.5)
            
            self.speed += (target_speed - self.speed) * dt * 2
            self.speed = max(0.0, min(1.0, self.speed))
        
        # Update rotation based on steering
        if self.speed > 0.05:
            max_turn_rate = 120
            turn_amount = self.steering * max_turn_rate * dt
            
            speed_factor = max(0.3, 1.0 - (self.speed * 0.7))
            turn_amount *= speed_factor
            
            self.rotation += turn_amount
        else:
            if abs(self.rotation) > 1.0:
                center_return = -self.rotation * 0.05
                self.rotation += center_return * dt * 60
        
        # Keep rotation in range
        self.rotation = self.rotation % 360
        if self.rotation > 180:
            self.rotation -= 360
        elif self.rotation < -180:
            self.rotation += 360

    def check_collision(self, obstacle_rect):
        """
        Check collision between the car and an obstacle rectangle.
        
        Args:
            obstacle_rect: pygame.Rect of the obstacle
        
        Returns:
            bool: True if collision detected
        """
        car_rect = pygame.Rect(
            self.x - self.width // 2,
            self.y - self.height // 2,
            self.width,
            self.height
        )
        return car_rect.colliderect(obstacle_rect)

    def draw(self, screen):
        """Draw the car on the screen"""
        try:
            # Create a surface for the car
            car_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            
            # Draw car body (blue rectangle with rounded corners)
            pygame.draw.rect(
                car_surface,
                BLUE,  # Using blue color defined at the top of the file
                (0, 0, self.width, self.height),
                0,  # Filled rectangle
                10  # Rounded corners
            )
            
            # Draw windshield
            windshield_width = self.width * 0.7
            windshield_height = self.height * 0.3
            pygame.draw.rect(
                car_surface,
                (150, 220, 255),  # Light blue windshield
                (
                    (self.width - windshield_width) / 2,
                    self.height * 0.15,
                    windshield_width,
                    windshield_height
                ),
                0,  # Fill rectangle
                5   # Slightly rounded corners
            )
            
            # Draw headlights
            light_size = self.width // 5
            # Left headlight
            pygame.draw.circle(
                car_surface,
                (255, 255, 200),  # Yellowish light
                (self.width // 4, light_size),
                light_size // 2
            )
            # Right headlight
            pygame.draw.circle(
                car_surface,
                (255, 255, 200),  # Yellowish light
                (self.width - self.width // 4, light_size),
                light_size // 2
            )
            
            # Draw brake lights if braking
            if self.braking:
                # Left brake light
                pygame.draw.circle(
                    car_surface,
                    (255, 0, 0),  # Red
                    (self.width // 4, self.height - light_size),
                    light_size // 2
                )
                # Right brake light
                pygame.draw.circle(
                    car_surface,
                    (255, 0, 0),  # Red
                    (self.width - self.width // 4, self.height - light_size),
                    light_size // 2
                )
                
            # Draw boost effect if boosting
            if self.boosting:
                flame_points = [
                    (self.width // 2, self.height),
                    (self.width // 2 - self.width // 4, self.height + self.height // 3),
                    (self.width // 2 + self.width // 4, self.height + self.height // 3)
                ]
                pygame.draw.polygon(car_surface, (255, 165, 0), flame_points)
                
            # Rotate the car surface
            rotated_car = pygame.transform.rotate(car_surface, -self.rotation)
            
            # Get the rect of the rotated car and position it
            rotated_rect = rotated_car.get_rect(center=(self.x, self.y))
            
            # Draw the rotated car
            screen.blit(rotated_car, rotated_rect)
            
        except Exception as e:
            print(f"Error drawing car: {e}")
            # Fallback to simple rectangle if rotation fails
            car_rect = pygame.Rect(
                self.x - self.width // 2,
                self.y - self.height // 2,
                self.width,
                self.height
            )
            pygame.draw.rect(screen, BLUE, car_rect)

class Obstacle:
    """Obstacle class for road hazards"""
    def __init__(self, x, y, speed=200):
        self.x = x
        self.y = y
        self.width = 30
        self.height = 40
        self.speed = speed
        self.hit = False
        self.color = (255, 140, 0)
        self.rect = pygame.Rect(x - self.width // 2, y - self.height // 2, 
                               self.width, self.height)
    
    def update(self, dt):
        """Update obstacle position"""
        self.y += self.speed * dt
        self.rect.y = self.y - self.height // 2
    
    def draw(self, screen):
        """Draw obstacle as traffic cone"""
        color = RED if self.hit else self.color
        
        points = [
            (self.x, self.y - self.height // 2),
            (self.x - self.width // 2, self.y + self.height // 2),
            (self.x + self.width // 2, self.y + self.height // 2)
        ]
        pygame.draw.polygon(screen, color, points)
        
        pygame.draw.line(screen, WHITE,
                        (self.x - self.width // 4, self.y),
                        (self.x + self.width // 4, self.y), 3)

class Game:
    """Main game class with steering-based background movement"""
    
    def __init__(self, mode="normal", hand_detector=None, show_tutorial=True, config=None):
        """
        Initialize the game
        
        Args:
            mode: str, game mode ("easy", "normal", "hard")
            hand_detector: HandDetector, optional hand detector
            show_tutorial: bool, show tutorial flag
            config: dict, optional configuration
        """
        # Initialize pygame
        pygame.init()
        
        # Screen setup
        self.screen_width = WINDOW_WIDTH
        self.screen_height = WINDOW_HEIGHT
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption(CAPTION)
        
        # Game settings from mode
        self.mode = mode
        self.settings = GAME_MODES.get(mode, GAME_MODES["normal"])
        
        # Timing
        self.clock = pygame.time.Clock()
        self._target_fps = FPS
        self.running = False
        
        # Game state
        self._paused = False
        self._game_over = False
        self._game_completed = False
        self._debug_mode = False
        self._debug_input = False
        self._show_help = False
        self.show_camera = True
        
        # Game timing
        self._start_time = 0
        self._elapsed_time = 0
        self._game_duration = self.settings["time_limit"]
        self._time_remaining = self._game_duration
        self.time_left = self._game_duration
        
        # Score and distance tracking
        self._score = 0
        self._distance_traveled = 0
        self._last_position = None
        
        # Car setup
        self._car = Car(self.screen_width // 2, self.screen_height - 100)
        
        # Obstacles
        self._obstacles = []
        self._next_obstacle_time = 0
        
        # Road animation
        self.road_offset = 0
        self.total_dash_cycle = 70  # 50 + 20
        self._road_offset_x = 0
        
        # Moving road
        if MOVING_ROAD_AVAILABLE:
            try:
                self._moving_road = MovingRoadGenerator(self.screen_width, self.screen_height)
                print("✅ MovingRoad initialized successfully")
            except Exception as e:
                print(f"⚠️ Error initializing MovingRoad: {e}")
                self._moving_road = None
        else:
            self._moving_road = None
        
        # Initialize fonts
        self._font = pygame.font.Font(None, 36)
        self._title_font = pygame.font.Font(None, 72)
        
        # Hand detection setup
        self._hand_detector = hand_detector or HandDetector()
        self._gesture_recognizer = GestureRecognizer()
        
        # Camera setup
        self._camera = self._init_camera(0, CAMERA_WIDTH, CAMERA_HEIGHT)
        self._frame = None
        self._last_gesture_result = None
        
        # Sound manager
        self._sound_manager = SoundManager()
        
        # UI buttons
        self.pause_button = Button(10, self.screen_height - 50, 80, 30, "Pause", PRIMARY, WHITE, self.toggle_pause)
        self.mute_button = Button(100, self.screen_height - 50, 80, 30, "Mute", SECONDARY, WHITE, self.toggle_mute)
        
        print(f"✅ Game initialized in {mode} mode")
        if self._hand_detector and self._hand_detector.enabled:
            print("✅ Hand detection enabled")
        else:
            print("⚠️ Hand detection disabled - using keyboard controls")
    
    def _init_camera(self, camera_index=0, width=640, height=480):
        """Initialize camera with robust error handling and fallback options
        
        Args:
            camera_index: int, camera device index
            width: int, camera width
            height: int, camera height
            
        Returns:
            cv2.VideoCapture or None on failure
        """
        try:
            if camera_index is None:
                camera_index = 0
            
            # Try different backends in order of preference
            backends = [
                cv2.CAP_DSHOW,      # DirectShow (Windows) - usually most reliable
                cv2.CAP_MSMF,       # Windows Media Foundation 
                cv2.CAP_ANY         # Let OpenCV choose
            ]
            
            for backend in backends:
                try:
                    print(f"🔍 Trying camera {camera_index} with backend {backend}")
                    camera = cv2.VideoCapture(camera_index, backend)
                    
                    if not camera.isOpened():
                        camera.release()
                        continue
                    
                    # Set camera properties
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    camera.set(cv2.CAP_PROP_FPS, 30)
                    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to prevent lag
                    
                    # Test if we can actually read a frame
                    ret, test_frame = camera.read()
                    if ret and test_frame is not None:
                        print(f"✅ Camera initialized successfully with backend {backend}")
                        return camera
                    else:
                        print(f"❌ Camera opened but cannot read frames with backend {backend}")
                        camera.release()
                        
                except Exception as e:
                    print(f"❌ Failed to initialize camera with backend {backend}: {e}")
                    continue
            
            # If all backends fail, try different camera indices
            for idx in range(3):  # Try camera indices 0, 1, 2
                if idx == camera_index:
                    continue  # Already tried this one
                    
                try:
                    print(f"🔍 Trying camera index {idx}")
                    camera = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                    
                    if camera.isOpened():
                        ret, test_frame = camera.read()
                        if ret and test_frame is not None:
                            print(f"✅ Found working camera at index {idx}")
                            camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            return camera
                    camera.release()
                    
                except Exception as e:
                    print(f"❌ Camera {idx} failed: {e}")
                    continue
            
            print(f"❌ Failed to open any camera")
            return None
            
        except Exception as e:
            print(f"❌ Error initializing camera: {e}")
            return None
    
    def read_frame(self):
        """
        Read frame from camera with robust error handling
        
        Returns:
            tuple: (success, frame)
        """
        if self._camera is None:
            return False, None
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                ret, frame = self._camera.read()
                
                if ret and frame is not None:
                    return True, frame
                else:
                    if attempt == 0:  # Only log on first failure
                        print(f"⚠️ Camera read failed, attempt {attempt + 1}/{max_retries}")
                    
                    # Try to restart camera on failure
                    if attempt < max_retries - 1:
                        self._restart_camera()
                        
            except Exception as e:
                print(f"⚠️ Error reading frame (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    self._restart_camera()
        
        return False, None
    
    def _restart_camera(self):
        """Restart the camera connection"""
        try:
            if self._camera:
                self._camera.release()
            
            # Wait a moment before reinitializing
            time.sleep(0.1)
            
            # Try to reinitialize camera
            self._camera = self._init_camera(0, CAMERA_WIDTH, CAMERA_HEIGHT)
            
            if self._camera:
                print("🔄 Camera restarted successfully")
            else:
                print("❌ Camera restart failed")
                
        except Exception as e:
            print(f"❌ Error restarting camera: {e}")
            self._camera = None

    def process_camera_input(self):
        """Process camera input for gesture detection with enhanced error handling"""
        if self._camera is None or self._hand_detector is None or not self._hand_detector.enabled:
            return None
        
        try:
            success, self._frame = self.read_frame()
            
            if not success or self._frame is None:
                # Return default gesture instead of None to keep game running
                return {
                    'steering': 0.0,
                    'throttle': 0.5,
                    'braking': False,
                    'boost': False,
                    'stop': False,
                    'gesture_name': 'Camera Error',
                    'confidence': 0.0,
                    'detection_fps': 0
                }
            
            self._frame = cv2.flip(self._frame, 1)
            self._frame, results = self._hand_detector.find_hands(self._frame)
            landmarks = self._hand_detector.find_positions(self._frame, results)
            
            if landmarks and self._gesture_recognizer:
                gestures = self._gesture_recognizer.recognize_gestures(landmarks, self._frame.shape[0])
                
                if gestures and gestures.get('gesture_name', 'No Detection') != 'No Detection':
                    current_time = time.time()
                    if not hasattr(self, '_last_gesture_log') or current_time - self._last_gesture_log > 1.0:
                        print(f"🤏 Gesture detected: {gestures.get('gesture_name', 'Unknown')}")
                        print(f"   Steering: {gestures.get('steering', 0):.2f}, Throttle: {gestures.get('throttle', 0):.2f}")
                        print(f"   Confidence: {gestures.get('confidence', 0):.2f}, FPS: {gestures.get('detection_fps', 0)}")
                        self._last_gesture_log = current_time
                
                return gestures
            else:
                return {
                    'steering': 0.0,
                    'throttle': 0.5,
                    'braking': False,
                    'boost': False,
                    'stop': False,
                    'gesture_name': 'No Hand Detected',
                    'confidence': 0.0,
                    'detection_fps': 0
                }
                
        except Exception as e:
            print(f"⚠️ Error in camera processing: {e}")
            
            # Try to restart camera on persistent errors
            if not hasattr(self, '_last_camera_restart') or time.time() - self._last_camera_restart > 5.0:
                self._last_camera_restart = time.time()
                self._restart_camera()
            
            # Return default gesture to keep game running
            return {
                'steering': 0.0,
                'throttle': 0.5,
                'braking': False,
                'boost': False,
                'stop': False,
                'gesture_name': 'Processing Error',
                'confidence': 0.0,
                'detection_fps': 0
            }

    def handle_input(self):
        """Handle keyboard input"""
        return pygame.key.get_pressed()

    def toggle_pause(self):
        """Toggle pause state"""
        self._paused = not self._paused
        print(f"Game {'paused' if self._paused else 'resumed'}")

    def toggle_mute(self):
        """Toggle mute state"""
        muted = self._sound_manager.toggle_mute()
        print(f"Sound {'muted' if muted else 'unmuted'}")
        return muted

    def update_obstacles(self, current_time, delta_time):
        """Update obstacles"""
        # Spawn new obstacles
        if current_time >= self._next_obstacle_time:
            # Create obstacle
            obstacle_x = random.randint(
                self.screen_width // 2 - 100,
                self.screen_width // 2 + 100
            )
            obstacle = Obstacle(obstacle_x, -50, self.settings["obstacle_speed"])
            self._obstacles.append(obstacle)
            
            # Schedule next obstacle
            frequency = self.settings["obstacle_frequency"]
            self._next_obstacle_time = current_time + random.uniform(1.0 / frequency, 3.0 / frequency)
        
        # Update existing obstacles
        for obstacle in self._obstacles[:]:
            obstacle.update(delta_time)  # Fixed: changed dt to delta_time
            
            # Remove obstacles that are off screen
            if obstacle.y > self.screen_height + 50:
                self._obstacles.remove(obstacle)
                # Award points for avoiding obstacle
                self._score += SCORE_PER_OBSTACLE * self.settings["score_multiplier"]

    def draw_built_in_road(self):
        """Draw built-in road animation"""
        # Fill with grass color
        self.screen.fill((0, 100, 0))
        
        # Draw road
        road_width = 300
        road_x = self.screen_width // 2 - road_width // 2
        pygame.draw.rect(self.screen, (80, 80, 80), (road_x, 0, road_width, self.screen_height))
        
        # Draw moving center line
        line_width = 6
        dash_length = 30
        gap_length = 20
        dash_spacing = dash_length + gap_length
        
        offset = int(self.road_offset) % dash_spacing
        
        y = -offset
        while y < self.screen_height + dash_spacing:
            if y + dash_length > 0:
                pygame.draw.rect(self.screen, WHITE,
                               (self.screen_width // 2 - line_width // 2, y,
                                line_width, min(dash_length, self.screen_height - y)))
            y += dash_spacing

    def draw_camera_feed(self):
        """Draw camera feed"""
        if self._frame is not None and self.show_camera:
            try:
                display_width = 320
                display_height = 240
                
                display_frame = cv2.resize(self._frame, (display_width, display_height))
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
                feed_surface = pygame.surfarray.make_surface(display_frame.swapaxes(0, 1))
                
                feed_rect = pygame.Rect(self.screen_width - display_width, 0, display_width, display_height)
                pygame.draw.rect(self.screen, WHITE, feed_rect, width=2)
                self.screen.blit(feed_surface, feed_rect)
                
                status_color = GREEN if (self._hand_detector and self._hand_detector.enabled) else RED
                status_text = "HAND DETECTION ACTIVE" if (self._hand_detector and self._hand_detector.enabled) else "HAND DETECTION DISABLED"
                
                font = pygame.font.SysFont(None, 16)
                status_surface = font.render(status_text, True, status_color)
                self.screen.blit(status_surface, (self.screen_width - display_width + 5, display_height + 5))
                
                if hasattr(self, '_last_gesture_result') and self._last_gesture_result:
                    gesture_name = self._last_gesture_result.get('gesture_name', 'Unknown')
                    gesture_surface = font.render(f"Gesture: {gesture_name}", True, WHITE)
                    self.screen.blit(gesture_surface, (self.screen_width - display_width + 5, display_height + 25))
                
            except Exception as e:
                logger.error(f"Error drawing camera feed: {e}")

    def draw_help(self):
        """Draw help screen"""
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        help_text = [
            "HAND GESTURE CONTROLS",
            "",
            "• Hand Tilt: Steering",
            "• Hand Height: Throttle",
            "• Fist: Brake",
            "• Thumbs Up: Boost",
            "• Open Palm: Stop",
            "",
            "KEYBOARD CONTROLS",
            "",
            "• Arrow Keys / WASD: Movement",
            "• Space: Boost",
            "• P: Pause",
            "• M: Mute",
            "• H: Help",
            "• ESC: Exit"
        ]
        
        y_offset = self.screen_height // 2 - len(help_text) * 15
        for line in help_text:
            if line.startswith("•"):
                color = (150, 150, 255)
            elif line == line.upper() and line:
                color = (255, 255, 0)
            else:
                color = WHITE
            
            text_surface = self._font.render(line, True, color)
            text_rect = text_surface.get_rect(center=(self.screen_width // 2, y_offset))
            self.screen.blit(text_surface, text_rect)
            y_offset += 30

    def handle_events(self):
        """Handle all events for the game"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_p:
                    self.toggle_pause()
                elif event.key == pygame.K_m:
                    self.toggle_mute()
                elif event.key == pygame.K_h:
                    self._show_help = not self._show_help
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    # Check if pause button is clicked
                    if self.pause_button.rect.collidepoint(event.pos):
                        self.toggle_pause()
                    # Check if mute button is clicked
                    elif self.mute_button.rect.collidepoint(event.pos):
                        self.toggle_mute()

    def cleanup(self):
        """Clean up game resources with enhanced camera cleanup"""
        try:
            if self._camera:
                print("🧹 Cleaning up camera...")
                self._camera.release()
                self._camera = None
                
            if self._hand_detector:
                print("🧹 Cleaning up hand detector...")
                self._hand_detector.release()
                
            self._sound_manager.cleanup()
            
            # Ensure all OpenCV windows are closed
            cv2.destroyAllWindows()
            
            # Wait a moment for cleanup
            time.sleep(0.1)
            
            pygame.quit()
            print("✅ Cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")
            # Force cleanup even if there are errors
            try:
                pygame.quit()
            except:
                pass

    def draw(self):
        """Draw the game screen with moving background"""
        self.screen.fill((50, 50, 50))
        
        # Draw moving road background
        if self._moving_road:
            try:
                self._moving_road.draw(self.screen)
            except Exception as e:
                print(f"⚠️ Error drawing moving road: {e}")
                self._draw_built_in_road()
        else:
            self._draw_built_in_road()
        
        # Calculate world offset for steering effect
        world_offset_x = int(self._road_offset_x)
        
        # Draw obstacles with world offset
        for obstacle in self._obstacles:
            if hasattr(obstacle, 'world_x'):
                screen_x = obstacle.world_x - world_offset_x
            else:
                obstacle.world_x = obstacle.x
                screen_x = obstacle.x - world_offset_x
                
            original_x = obstacle.x
            obstacle.x = screen_x
            obstacle.draw(self.screen)
            obstacle.x = original_x
        
        # Draw car (always in center)
        self._car.draw(self.screen)
        
        # Draw UI elements
        self._draw_ui()
        self._draw_camera_feed()
        
        if self._paused:
            self._draw_pause()
        
        if self._game_over or self._game_completed:
            self._draw_game_over()
            
        if self._debug_mode:
            self._draw_debug()
        
        if self._show_help:
            self._draw_help()
            
        pygame.display.flip()

    def update(self, delta_time):
        """
        Update game state with enhanced road movement
        
        Args:
            delta_time: float, time since last update
        """
        if self._paused or self._game_over:
            return
        
        # Update timing
        current_time = time.time()
        if self._start_time == 0:
            self._start_time = current_time
        
        self._elapsed_time = current_time - self._start_time
        self._time_remaining = max(0, self._game_duration - self._elapsed_time)
        
        if self.settings["time_limit"] > 0:
            elapsed = current_time - self._start_time
            self.time_left = max(0, self.settings["time_limit"] - elapsed)
            if self.time_left <= 0:
                self._game_over = True
                logger.info("Game over: Time's up")
                return
        
        if self._elapsed_time >= self._game_duration and not self._game_completed:
            self._game_completed = True
            print(f"Game completed! Final score: {int(self._score)}")
        
        # Process input
        gestures = self._process_camera_input()
        if gestures:
            self._last_gesture_result = gestures
        
        keys = self._handle_input()
        
        controls = {
            'steering': 0.0,
            'throttle': 0.5,
            'braking': False,
            'boost': False
        }
        
        if gestures and self._hand_detector and self._hand_detector.enabled:
            if gestures.get('stop', False):
                controls['steering'] = 0.0
                controls['throttle'] = 0.0
                controls['braking'] = True
            elif gestures.get('braking', False):
                controls['braking'] = True
                controls['throttle'] = 0.0
                controls['steering'] = gestures.get('steering', 0.0) * 0.5
            else:
                controls['steering'] = gestures.get('steering', 0.0)
                controls['throttle'] = gestures.get('throttle', 0.5)
                controls['boost'] = gestures.get('boost', False)
        
        # Keyboard overrides
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            controls['steering'] = -1.0
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            controls['steering'] = 1.0
        
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            controls['throttle'] = 1.0
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            controls['throttle'] = 0.0
            controls['braking'] = True
        
        if keys[pygame.K_SPACE]:
            controls['boost'] = True
        
        # Update car
        self._car.update(controls, delta_time)
        
        # ENHANCED: Update road animation with stronger steering effect
        road_speed = abs(self._car.speed) * 400  # Increased from 300 to 400
        self.road_offset += road_speed * delta_time
        
        # Enhanced steering-based horizontal movement
        steering_factor = self._car.rotation / 15.0  # More sensitive (15 instead of 20)
        steering_factor = max(-3.0, min(3.0, steering_factor))  # Allow more extreme values
        horizontal_speed = steering_factor * 250 * self._car.speed  # Increased from 180 to 250
        self._road_offset_x += horizontal_speed * delta_time
        
        # Reset road offset to prevent overflow
        if self.road_offset >= self.total_dash_cycle:
            self.road_offset -= self.total_dash_cycle
        
        # Update moving road with enhanced parameters
        if self._moving_road:
            try:
                self._moving_road.update(self._car.rotation, self._car.speed, delta_time)
            except Exception as e:
                logger.error(f"Error updating moving road: {e}")
        
        # Update score
        current_position = (self._car.x, self._car.y)
        if self._last_position:
            distance = math.hypot(current_position[0] - self._last_position[0],
                                current_position[1] - self._last_position[1])
            self._distance_traveled += distance
            self._score = self._distance_traveled / 10
        self._last_position = current_position
        
        self._update_obstacles(current_time, delta_time)
        self._check_collisions()

    def run(self):
        """Main game loop with enhanced road animation"""
        self.running = True
        last_time = time.time()
        
        logger.info("Starting game loop")
        
        self._start_time = time.time()
        self._last_position = (self._car.x, self._car.y)
        
        try:
            while self.running:
                current_time = time.time()
                dt = min(current_time - last_time, 0.1)
                last_time = current_time
                
                self.handle_events()
                
                self.update(dt)
                
                self.draw()
                
                self.clock.tick(self._target_fps)
                
        except Exception as e:
            logger.error(f"Error in game loop: {e}")
            raise
        finally:
            self.cleanup()

    def _get_default_gestures(self):
        """Return default gesture values"""
        return {
            'steering': 0.0,
            'throttle': 0.5,
            'braking': False,
            'boost': False,
            'stop': False,
            'confidence': 0.0,
            'detection_fps': 0,
            'gesture_name': 'No Hand Detected'
        }
    
    def _process_camera_input(self):
        """Process camera input for gesture detection"""
        return self.process_camera_input()

    def _handle_input(self):
        """Handle keyboard input"""
        return self.handle_input()

    def _update_obstacles(self, current_time, delta_time):
        """Update obstacles"""
        return self.update_obstacles(current_time, delta_time)

    def _check_collisions(self):
        """Check for collisions between car and obstacles"""
        for obstacle in self._obstacles[:]:  # Use slice copy to safely remove during iteration
            if self._car.check_collision(obstacle.rect):
                # Instead of immediate game over, remove the obstacle and continue
                self._obstacles.remove(obstacle)
                print("💥 Collision detected! Obstacle removed.")
                # Optionally reduce score or health here
                self._score = max(0, self._score - 50)  # Penalty instead of game over
                break

    def _draw_built_in_road(self):
        """Draw built-in road animation"""
        return self.draw_built_in_road()

    def _draw_camera_feed(self):
        """Draw camera feed"""
        return self.draw_camera_feed()

    def _draw_help(self):
        """Draw help screen"""
        return self.draw_help()

    def _draw_ui(self):
        """Draw UI elements"""
        # Draw score
        draw_text(self.screen, f"Score: {int(self._score)}", WHITE, self._font, 20, 20)
        
        # Draw time remaining
        minutes = int(self._time_remaining) // 60
        seconds = int(self._time_remaining) % 60
        time_text = f"Time: {minutes}:{seconds:02d}"
        draw_text(self.screen, time_text, WHITE, self._font, self.screen_width - 150, 20)
        
        # Draw speed indicator
        speed_text = f"Speed: {self._car.speed:.1f}"
        draw_text(self.screen, speed_text, WHITE, self._font, 20, 60)
        
        # Draw controls info
        if self._last_gesture_result:
            gesture_text = f"Gesture: {self._last_gesture_result.get('gesture_name', 'Unknown')}"
            draw_text(self.screen, gesture_text, GREEN, self._font, 20, 100)
        
        # Draw buttons
        self.pause_button.draw(self.screen)
        self.mute_button.draw(self.screen)

    def _draw_pause(self):
        """Draw pause overlay"""
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.set_alpha(128)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        pause_text = "PAUSED"
        text_surface = self._title_font.render(pause_text, True, WHITE)
        text_rect = text_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        self.screen.blit(text_surface, text_rect)
        
        instruction_text = "Press P to resume"
        instruction_surface = self._font.render(instruction_text, True, WHITE)
        instruction_rect = instruction_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 60))
        self.screen.blit(instruction_surface, instruction_rect)

    def _draw_game_over(self):
        """Draw game over screen"""
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.set_alpha(180)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        if self._game_completed:
            title_text = "GAME COMPLETED!"
            title_color = SUCCESS
        else:
            title_text = "GAME OVER"
            title_color = ERROR
            
        title_surface = self._title_font.render(title_text, True, title_color)
        title_rect = title_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 60))
        self.screen.blit(title_surface, title_rect)
        
        score_text = f"Final Score: {int(self._score)}"
        score_surface = self._font.render(score_text, True, WHITE)
        score_rect = score_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        self.screen.blit(score_surface, score_rect)
        
        instruction_text = "Press ESC to exit"
        instruction_surface = self._font.render(instruction_text, True, WHITE)
        instruction_rect = instruction_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 60))
        self.screen.blit(instruction_surface, instruction_rect)

    def _draw_debug(self):
        """Draw debug information"""
        if not self._debug_mode:
            return
            
        debug_y = 200
        debug_texts = [
            f"Car Position: ({self._car.x:.1f}, {self._car.y:.1f})",
            f"Car Rotation: {self._car.rotation:.1f}°",
            f"Car Speed: {self._car.speed:.2f}",
            f"Road Offset: {self.road_offset:.1f}",
            f"Road Offset X: {self._road_offset_x:.1f}",
            f"Obstacles: {len(self._obstacles)}",
            f"FPS: {self.clock.get_fps():.1f}"
        ]
        
        for i, text in enumerate(debug_texts):
            draw_text(self.screen, text, WHITE, self._font, 20, debug_y + i * 25)

    def check_game_over_conditions(self):
        """בדיקת תנאי סיום המשחק - יותר סלחנית"""
        
        # בדיקת התנגשות עם מכשולים - רק התנגשות ישירה
        for obstacle in self.obstacles:
            distance = math.sqrt((self.car.x - obstacle.x)**2 + (self.car.y - obstacle.y)**2)
            
            # הגדל את הסובלנות להתנגשות
            collision_threshold = 40  # במקום 30
            
            if distance < collision_threshold:
                self.collision_count += 1
                
                # אפשר יותר התנגשויות לפני סיום המשחק
                if self.collision_count >= 3:  # במקום 1
                    return True
                
                # הסר את המכשול שנגענו בו
                self.obstacles.remove(obstacle)
                break
        
        # בדיקת יציאה מהמסך - יותר סלחנית
        screen_margin = 100  # במקום 50
        
        if (self.car.x < -screen_margin or 
            self.car.x > self.screen_width + screen_margin or
            self.car.y < -screen_margin or 
            self.car.y > self.screen_height + screen_margin):
            
            # התחזר למקום בטוח במקום לסיים את המשחק
            self.car.x = max(screen_margin, min(self.screen_width - screen_margin, self.car.x))
            self.car.y = max(screen_margin, min(self.screen_height - screen_margin, self.car.y))
            
            return False  # לא מסיים את המשחק
        
        return False