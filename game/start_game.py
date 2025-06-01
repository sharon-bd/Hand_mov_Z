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
            
            # Calculate hand center and orientation
            hand_center_x = (wrist[0] + index_mcp[0]) / 2
            hand_center_y = (wrist[1] + index_mcp[1]) / 2
            
            # 1. STEERING - Hand tilt for left/right
            steering = self._calculate_steering(landmarks, wrist, index_mcp)
            
            # 2. THROTTLE - Hand height for speed
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
        """Calculate throttle based on hand height"""
        try:
            # Normalize hand height (0 = top of frame, 1 = bottom of frame)
            normalized_height = hand_center_y / frame_height
            
            # Invert so higher hand = more throttle
            # Map to 0.0-1.0 range with sensitivity adjustment
            throttle = (1.0 - normalized_height) * self.throttle_sensitivity
            throttle = max(0.0, min(1.0, throttle))
            
            return throttle
            
        except Exception:
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
            return 'Neutral'
    
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
    
    def _init_camera(self, camera_index=0, width=640, height=480):
        """Initialize camera with specified settings
        
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
                
            camera = cv2.VideoCapture(camera_index)
            if not camera.isOpened():
                print(f"❌ Failed to open camera at index {camera_index}")
                return None
                
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            print(f"✅ Camera initialized at index {camera_index}")
            return camera
        except Exception as e:
            print(f"❌ Error initializing camera: {e}")
            return None
    
    def __init__(self, mode="normal", screen_width=800, screen_height=600, hand_detector=None):
        """
        Initialize the game instance
        
        Args:
            mode: str, game mode ("normal", "easy", "hard")
            screen_width: int, screen width
            screen_height: int, screen height
            hand_detector: HandDetector, optional hand detector instance
        """
        self.mode = mode
        self.settings = GAME_MODES.get(mode, GAME_MODES["normal"])
        
        # Game state
        self.running = False
        self._paused = False
        self._game_over = False
        
        # Score and time
        self._score = 0.0
        self.time_left = self.settings["time_limit"]
        self._start_time = 0
        
        # Screen dimensions
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Road animation properties
        self.road_offset = 0
        self._road_offset_x = 0
        self.dash_length = 30
        self.gap_length = 20
        self.total_dash_cycle = self.dash_length + self.gap_length  # Fixed: changed _gap_length to gap_length
        
        # Initialize pygame
        if not pygame.get_init():
            pygame.init()
        
        # Display setup
        self.screen = pygame.display.set_mode((self.screen_width + 320, self.screen_height))
        pygame.display.set_caption(f"{CAPTION} - {mode.capitalize()}")
        self.clock = pygame.time.Clock()
        self._target_fps = FPS
        
        # Game objects
        self._car = Car(self.screen_width // 2, self.screen_height - 100)
        self._obstacles = []
        self._next_obstacle_time = 0
        
        # Moving road setup
        self._moving_road = None
        if MOVING_ROAD_AVAILABLE:
            try:
                self._moving_road = MovingRoadGenerator(self.screen_width, self.screen_height)  # Fixed: changed _screen_height to screen_height
            except Exception as e:
                print(f"Error initializing MovingRoadGenerator: {e}")
                self._moving_road = None
        
        # Camera and gesture detection
        self._camera = None
        self._frame = None
        self._hand_detector = hand_detector
        self._gesture_recognizer = None
        self.init_camera_and_detector()  # Fixed: changed _init_camera_and_detector to init_camera_and_detector
        
        # Sound system
        self._sound_manager = SoundManager()
        
        # Initialize fonts
        self._font = pygame.font.Font(None, 36)
        self._title_font = pygame.font.Font(None, 72)
        
        # Initialize debug and display flags
        self._debug_mode = False
        self._debug_input = False
        self._show_help = False
        self.show_camera = True
        self._last_debug_time = 0
        self._last_gesture_result = None
        self._game_completed = False
        self._game_duration = 300  # 5 minutes
        self._elapsed_time = 0
        self._time_remaining = 0
        self._distance_traveled = 0
        self._last_position = None
        
        # Create UI elements
        self.create_ui_elements()

    def init_camera_and_detector(self):
        """Initialize camera and hand detection"""
        try:
            camera_index = None
            if 'SELECTED_CAMERA' in os.environ:
                try:
                    camera_index = int(os.environ['SELECTED_CAMERA'])
                except ValueError:
                    pass
            
            self._camera = self._init_camera(camera_index, CAMERA_WIDTH, CAMERA_HEIGHT)
            
            if self._camera is None:
                logger.error("Failed to initialize camera")
                print("❌ Camera initialization failed")
            else:
                print("✅ Camera initialized successfully")
                
                self._hand_detector = HandDetector()
                self._gesture_recognizer = GestureRecognizer()
                
                if self._hand_detector.enabled:
                    logger.info("✅ Real hand detection system initialized")
                    print("✅ Hand gesture detection is ACTIVE")
                else:
                    logger.warning("⚠️ Hand detection disabled - using keyboard only")
                    print("⚠️ Hand detection disabled - using keyboard controls only")
                
        except Exception as e:
            logger.error(f"Error initializing camera/detector: {e}")
            print(f"❌ Error initializing camera/detector: {e}")
            self._camera = None
            self._hand_detector = None
            self._gesture_recognizer = None

    def create_ui_elements(self):
        """Create UI buttons and elements"""
        self.pause_button = Button(
            self.screen_width - 110, 10, 100, 40, "Pause", 
            PRIMARY, ACCENT, self.toggle_pause
        )
        
        self.mute_button = Button(
            self.screen_width - 110, 60, 100, 40, "Mute", 
            PRIMARY, ACCENT, self.toggle_mute
        )

    def read_frame(self):
        """
        Read frame from camera
        
        Returns:
            tuple: (success, frame)
        """
        if self._camera is None:
            return False, None
        
        try:
            ret, frame = self._camera.read()
            return ret, frame
        except Exception as e:
            print(f"Error reading frame: {e}")
            return False, None
    
    def create_progress_bar(self, width, height, progress, color):
        """
        Create a progress bar surface
        
        Args:
            width: int, bar width
            height: int, bar height
            progress: float, progress value (0-1)
            color: tuple, fill color
            
        Returns:
            pygame.Surface bar
        """
        bar_surface = pygame.Surface((width, height))
        bar_surface.fill((100, 100, 100))
        
        fill_width = int(width * progress)
        if fill_width > 0:
            pygame.draw.rect(bar_surface, color, (0, 0, fill_width, height))
        
        return bar_surface
    
    def toggle_pause(self):
        """Toggle game pause state"""
        self._paused = not self._paused
        logger.info(f"Game {'paused' if self._paused else 'resumed'}")

    def toggle_mute(self):
        """Toggle sound mute state"""
        try:
            is_muted = self._sound_manager.toggle_mute()
            logger.info(f"Sound {'muted' if is_muted else 'unmuted'}")
        except Exception as e:
            logger.error(f"Error toggling mute: {e}")

    def handle_input(self):
        """
        Handle keyboard input with debug output
        
        Returns:
            dict: Pressed keys
        """
        keys = pygame.key.get_pressed()
        current_time = time.time()
        
        if self._debug_input and current_time - self._last_debug_time > 1:
            any_key_pressed = any([
                keys[pygame.K_UP],
                keys[pygame.K_DOWN],
                keys[pygame.K_LEFT],
                keys[pygame.K_RIGHT],
                keys[pygame.K_w],
                keys[pygame.K_s],
                keys[pygame.K_a],
                keys[pygame.K_d],
                keys[pygame.K_SPACE]
            ])
            
            if any_key_pressed:
                print(f"Input: UP={keys[pygame.K_UP]}, DOWN={keys[pygame.K_DOWN]}, "
                      f"LEFT={keys[pygame.K_LEFT]}, RIGHT={keys[pygame.K_RIGHT]}")
                print(f"Car: speed={self._car.speed:.2f}, steering={self._car.steering:.2f}")
            
            self._last_debug_time = current_time
        
        return keys

    def draw_help(self):
        """Draw help text"""
        help_rect = pygame.Rect(10, 10, 200, 240)
        pygame.draw.rect(self.screen, (30, 30, 50), help_rect)
        pygame.draw.rect(self.screen, WHITE, help_rect, 2)
        
        help_lines = [
            "Controls:",
            "Arrow Keys: Steer/Speed",
            "WASD: Alternative",
            "Space: Boost",
            "H: Toggle Help",
            "D: Toggle Debug",
            "C: Toggle Camera",
            "M: Mute/Unmute",
            "P: Pause",
            "ESC: Exit"
        ]
        
        for i, line in enumerate(help_lines):
            color = ACCENT if i == 0 else WHITE
            help_text = pygame.font.Font(None, 20).render(line, True, color)
            self.screen.blit(help_text, (help_rect.x + 10, help_rect.y + 10 + i * 22))

    def draw_built_in_road(self):
        """Draw the built-in road with steering-based horizontal movement"""
        # Road surface
        road_color = (80, 80, 80)
        road_width = self.screen_width - 100
        
        # Calculate road position with enhanced horizontal offset
        road_x = 50 - int(self._road_offset_x * 1)
        
        pygame.draw.rect(self.screen, road_color, (road_x, 0, road_width, self.screen_height))
        
        # Moving center line
        center_x = road_x + road_width // 2
        line_width = 5
        
        offset = int(-self.road_offset) % (self.dash_length + self.gap_length)
        
        y = -offset
        while y < self.screen_height:
            if y + self.dash_length > 0:
                dash_start = max(0, y)
                dash_end = min(self.screen_height, y + self.dash_length)
                pygame.draw.rect(self.screen, WHITE,
                                (center_x - line_width // 2, dash_start,
                                 line_width, dash_end - dash_start))
            y += self.dash_length + self.gap_length
        
        # Edge lines
        edge_color = (255, 255, 0)
        pygame.draw.rect(self.screen, edge_color, (road_x, 0, 5, self.screen_height))
        pygame.draw.rect(self.screen, edge_color, (road_x + road_width - 5, 0, 5, self.screen_height))

    def update_obstacles(self, current_time, delta_time):
        """Update obstacles with world offset - only when car is moving"""
        # Don't spawn or move obstacles if the car is stopped
        if self._car.speed <= 0.05:  # Add a small threshold for determining "stopped"
            return
            
        if self.settings["obstacle_frequency"] > 0:
            if current_time >= self.next_obstacle_time:
                obstacle_world_x = random.randint(100, self.screen_width - 100) + self.road_offset_x
                obstacle = Obstacle(obstacle_world_x, -50, speed=self.settings["obstacle_speed"])
                obstacle.world_x = obstacle_world_x
                self.obstacles.append(obstacle)
                
                self.next_obstacle_time = current_time + (1.0 / self.settings["obstacle_frequency"])
        
        for obstacle in self.obstacles[:]:
            obstacle.update(delta_time)
            
            if not hasattr(obstacle, 'world_x'):
                obstacle.world_x = obstacle.x
            
            screen_x = obstacle.world_x - self.road_offset_x
            if obstacle.y > self.screen_height + 50 or screen_x < -150 or screen_x > self.screen_width + 150:
                self.obstacles.remove(obstacle)
                
                if not obstacle.hit:
                    points = SCORE_PER_OBSTACLE * self.settings["score_multiplier"]
                    self.score += points

    def process_camera_input(self):
        """Process camera input for gesture detection"""
        if self.camera is None or self.hand_detector is None or not self.hand_detector.enabled:
            return None
        
        try:
            success, self.frame = self.read_frame()
            
            if not success or self.frame is None:
                return None
            
            self.frame = cv2.flip(self.frame, 1)
            self.frame, results = self.hand_detector.find_hands(self.frame)
            landmarks = self.hand_detector.find_positions(self.frame, results)
            
            if landmarks and self.gesture_recognizer:
                gestures = self.gesture_recognizer.recognize_gestures(landmarks, self.frame.shape[0])
                
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
            logger.error(f"Error processing camera input: {e}")
            print(f"⚠️ Error in camera processing: {e}")
            return None

    def draw_camera_feed(self):
        """Draw camera feed display"""
        if self.frame is not None and self.show_camera:
            try:
                display_width = 320
                display_height = 240
                
                display_frame = cv2.resize(self.frame, (display_width, display_height))
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
                feed_surface = pygame.surfarray.make_surface(display_frame.swapaxes(0, 1))
                
                feed_rect = pygame.Rect(self.screen_width, 0, display_width, display_height)
                pygame.draw.rect(self.screen, WHITE, feed_rect, width=2)
                self.screen.blit(feed_surface, feed_rect)
                
                status_color = GREEN if (self.hand_detector and self.hand_detector.enabled) else RED
                status_text = "HAND DETECTION ACTIVE" if (self.hand_detector and self.hand_detector.enabled) else "HAND DETECTION DISABLED"
                
                font = pygame.font.SysFont(None, 16)
                status_surface = font.render(status_text, True, status_color)
                self.screen.blit(status_surface, (self.screen_width + 5, display_height + 5))
                
                if hasattr(self, '_last_gesture_result') and self._last_gesture_result:
                    gesture_name = self._last_gesture_result.get('gesture_name', 'Unknown')
                    gesture_surface = font.render(f"Gesture: {gesture_name}", True, WHITE)
                    self.screen.blit(gesture_surface, (self.screen_width + 5, display_height + 25))
                
            except Exception as e:
                logger.error(f"Error drawing camera feed: {e}")
                
        elif self.show_camera:
            camera_rect = pygame.Rect(self.screen_width, 0, 320, 300)
            pygame.draw.rect(self.screen, (30, 30, 30), camera_rect)
            
            messages = [
                "Camera Not Available",
                "Check camera connection",
                "",
                "Using Keyboard Controls:",
                "Arrow Keys / WASD: Movement",
                "Space: Boost",
                "",
                "Gesture Controls Ready:",
                "• Hand Tilted: Steering",
                "• Hand Height: Throttle",
                "• Fist: Brake",
                "• Thumbs Up: Boost",
                "• Open Palm: Stop",
                "",
                "MediaPipe READY ✅" if MEDIAPIPE_AVAILABLE else "KEYBOARD MODE ⌨️"
            ]
            
            for i, msg in enumerate(messages):
                if not msg:
                    continue
                    
                font_size = 16 if i > 0 else 18
                font = pygame.font.SysFont(None, font_size)
                
                if i == 0:
                    color = (255, 50, 50) if not MEDIAPIPE_AVAILABLE else (255, 165, 0)
                elif "MediaPipe READY" in msg:
                    color = (0, 255, 0)
                elif "KEYBOARD MODE" in msg:
                    color = (255, 255, 0)
                elif "Gesture Controls" in msg:
                    color = (255, 215, 0)
                elif msg.startswith("•"):
                    color = (150, 150, 255)
                else:
                    color = (200, 200, 200)
                    
                text = font.render(msg, True, color)
                self.screen.blit(text, (self.screen_width + 10, 50 + i * 18))

    def cleanup(self):
        """Clean up game resources"""
        if self._camera:
            self._camera.release()
        if self._hand_detector:
            self._hand_detector.release()
        self._sound_manager.cleanup()
        cv2.destroyAllWindows()
        pygame.quit()

    def draw(self):
        """Draw the game screen"""
        self.screen.fill((50, 50, 50))
        
        if self._moving_road:
            try:
                self._moving_road.draw(self.screen)
            except:
                self._draw_built_in_road()
        else:
            self._draw_built_in_road()
        
        world_offset_x = int(self._road_offset_x)
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
        
        self._car.draw(self.screen)
        
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

    def draw_ui(self):
        """Draw UI elements"""
        if hasattr(self, 'pause_button'):
            self.pause_button.draw(self.screen)
        if hasattr(self, 'mute_button'):
            self.mute_button.draw(self.screen)
        
        score_text = self._font.render(f"Score: {int(self._score)}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        
        speed_text = self._font.render(f"Speed: {self._car.speed:.2f}", True, WHITE)
        self.screen.blit(speed_text, (10, 40))
        
        if self.time_left > 0:
            minutes = int(self.time_left // 60)
            seconds = int(self.time_left % 60)
            time_text = self._font.render(f"Time: {minutes}:{seconds:02d}", True, WHITE)
            self.screen.blit(time_text, (self.screen_width // 2 - 50, 10))

    def draw_pause(self):
        """Draw pause menu"""
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        pause_text = self._title_font.render("PAUSED", True, WHITE)
        text_rect = pause_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        self.screen.blit(pause_text, text_rect)
        
        inst_text = self._font.render("Press P to resume", True, WHITE)
        inst_rect = inst_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 50))
        self.screen.blit(inst_text, inst_rect)

    def draw_game_over(self):
        """Draw game over screen"""
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        title = "Game Completed!" if self._game_completed else "Game Over"
        color = SUCCESS if self._game_completed else RED
        title_text = self._title_font.render(title, True, color)
        title_rect = title_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 50))
        self.screen.blit(title_text, title_rect)
        
        score_text = self._font.render(f"Final Score: {int(self._score)}", True, WHITE)
        score_rect = score_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        self.screen.blit(score_text, score_rect)
        
        inst_text = self._font.render("Press ESC to exit", True, WHITE)
        inst_rect = inst_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 50))
        self.screen.blit(inst_text, inst_rect)

    def draw_debug(self):
        """Draw debug information"""
        stats = self._gesture_recognizer.get_detection_stats() if self._gesture_recognizer else {}
        
        debug_info = [
            f"Car Pos: ({self._car.x:.1f}, {self._car.y:.1f})",
            f"Car Rotation: {self._car.rotation:.1f}°",
            f"Road Offset X: {self._road_offset_x:.1f}",
            f"Obstacles: {len(self._obstacles)}",
            f"FPS: {self.clock.get_fps():.1f}",
            "",
            f"Gesture FPS: {stats.get('fps', 0)}",
            f"Gesture History: {stats.get('history_size', 0)}",
            f"Smoothing: {'ON' if stats.get('smoothing_enabled', False) else 'OFF'}"
        ]
        
        for i, info in enumerate(debug_info):
            if not info:
                continue
            debug_text = self._font.render(info, True, WHITE)
            self.screen.blit(debug_text, (10, self.screen_height - 100 + i * 20))

    def confirm_exit(self):
        """
        Confirm exit intention
        
        Returns:
            bool: True to confirm exit
        """
        return True

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                print(f"🔑 Key pressed: {pygame.key.name(event.key)}")
                
                if event.key == pygame.K_ESCAPE:
                    if self.game_completed or self._confirm_exit():
                        self.running = False
                elif event.key == pygame.K_m:
                    self.toggle_mute()
                elif event.key == pygame.K_h:
                    self.show_help = not self.show_help
                elif event.key == pygame.K_d:
                    self.debug_mode = not self.debug_mode
                elif event.key == pygame.K_i:
                    self.debug_input = not self.debug_input
                    print(f"🔧 Input debug {'enabled' if self.debug_input else 'disabled'}")
                elif event.key == pygame.K_c:
                    self.show_camera = not self.show_camera
                elif event.key == pygame.K_p:
                    self.toggle_pause()
            
            if hasattr(self, 'pause_button'):
                self.pause_button.handle_event(event)
            if hasattr(self, 'mute_button'):
                self.mute_button.handle_event(event)

    def update(self, delta_time):
        """
        Update game state
        
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
        
        self._car.update(controls, delta_time)
        
        # Update road animation
        road_speed = abs(self._car.speed) * 300
        self.road_offset += road_speed * delta_time
        
        steering_factor = self._car.rotation / 20.0
        steering_factor = max(-2.5, min(2.5, steering_factor))
        horizontal_speed = steering_factor * 180 * self._car.speed
        self._road_offset_x += horizontal_speed * delta_time
        
        if self.road_offset >= self.total_dash_cycle:
            self.road_offset -= self.total_dash_cycle
        
        if self._moving_road:
            try:
                self._moving_road.update(self._car.rotation, self._car.speed, delta_time)
            except:
                logger.error(f"Error updating moving road")
        
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
    
    def check_collisions(self):
        """Check collisions between car and obstacles"""
        for obstacle in self.obstacles:
            if not obstacle.hit:
                world_x = getattr(obstacle, 'world_x', obstacle.x)
                screen_x = world_x - self.road_offset_x
                
                original_x = obstacle.x
                obstacle.x = screen_x
                
                if self.car.check_collision(obstacle.rect):
                    obstacle.hit = True
                    try:
                        self.sound_manager.play("collision")
                    except Exception as e:
                        logger.error(f"Error playing collision sound: {e}")
                    logger.info("Collision detected")
                    
                    if self.mode == "hard":
                        self.game_over = True
                        logger.info("Game over: Collision in hard mode")
                
                obstacle.x = original_x

    # Add missing property accessors and delegate methods
    @property
    def game_completed(self):
        return self._game_completed
    
    @property
    def debug_mode(self):
        return self._debug_mode
    
    @debug_mode.setter
    def debug_mode(self, value):
        self._debug_mode = value
    
    @property
    def debug_input(self):
        return self._debug_input
    
    @debug_input.setter
    def debug_input(self, value):
        self._debug_input = value
    
    @property
    def show_help(self):
        return self._show_help
    
    @show_help.setter
    def show_help(self, value):
        self._show_help = value
    
    @property
    def obstacles(self):
        return self._obstacles
    
    @property
    def next_obstacle_time(self):
        return self._next_obstacle_time
    
    @next_obstacle_time.setter
    def next_obstacle_time(self, value):
        self._next_obstacle_time = value
    
    @property
    def road_offset_x(self):
        return self._road_offset_x
    
    @property
    def car(self):
        return self._car
    
    @property
    def score(self):
        return self._score
    
    @score.setter
    def score(self, value):
        self._score = value
    
    @property
    def camera(self):
        return self._camera
    
    @property
    def hand_detector(self):
        return self._hand_detector
    
    @hand_detector.setter
    def hand_detector(self, value):
        self._hand_detector = value
    
    @property
    def gesture_recognizer(self):
        return self._gesture_recognizer
    
    @gesture_recognizer.setter
    def gesture_recognizer(self, value):
        self._gesture_recognizer = value
    
    @property
    def frame(self):
        return self._frame
    
    @frame.setter
    def frame(self, value):
        self._frame = value
    
    @property
    def sound_manager(self):
        return self._sound_manager
    
    @property
    def game_over(self):
        return self._game_over
    
    @game_over.setter
    def game_over(self, value):
        self._game_over = value
    
    # Delegate methods
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
        """Check collisions"""
        return self.check_collisions()
    
    def _draw_built_in_road(self):
        """Draw built-in road"""
        return self.draw_built_in_road()
    
    def _draw_ui(self):
        """Draw UI elements"""
        return self.draw_ui()
    
    def _draw_camera_feed(self):
        """Draw camera feed"""
        return self.draw_camera_feed()
    
    def _draw_pause(self):
        """Draw pause screen"""
        return self.draw_pause()
    
    def _draw_game_over(self):
        """Draw game over screen"""
        return self.draw_game_over()
    
    def _draw_debug(self):
        """Draw debug info"""
        return self.draw_debug()
    
    def _draw_help(self):
        """Draw help"""
        return self.draw_help()
    
    def _confirm_exit(self):
        """Confirm exit"""
        return self.confirm_exit()

    def run(self):
        """Main game loop"""
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

def run_game(mode="normal", hand_detector=None, show_tutorial=True, config=None):
    """
    Run the game with specified settings
    
    Args:
        mode: str, game mode ("easy", "normal", "hard")
        hand_detector: HandDetector, optional hand detector
        show width: bool, show tutorial flag
        config: dict, optional configuration
        
    Returns:
        float: Final score
    """
    print(f"Starting game in {mode} mode")
    
    game = Game(mode, hand_detector=hand_detector)
    
    try:
        game.run()
        return game.score
    except:
        return 0
    finally:
        game.cleanup()

__all__ = ['Game', 'run_game']