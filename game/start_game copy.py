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
    "practice": {
        "time_limit": 0,  # No time limit
        "obstacle_frequency": 0.0,  # No obstacles
        "obstacle_speed": 0,
        "score_multiplier": 0.5
    },
    "normal": {
        "time_limit": 180,
        "obstacle_frequency": 0.02,
        "obstacle_speed": 200,
        "score_multiplier": 1.0
    },
    "easy": {
        "time_limit": 240,
        "obstacle_frequency": 0.01,  # FIXED: Reduced frequency to spawn fewer turtles
        "obstacle_speed": 150,
        "score_multiplier": 0.8
    },
    "hard": {
        "time_limit": 120,
        "obstacle_frequency": 0.03,
        "obstacle_speed": 250,
        "score_multiplier": 1.5
    },
    "time_trial": {
        "time_limit": 60,  # Short time limit
        "obstacle_frequency": 0.04,  # More obstacles
        "obstacle_speed": 300,
        "score_multiplier": 2.0
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
            
            # PRIORITY 1: STOP - Open palm detection (highest priority)
            stop = self._detect_open_palm(landmarks, thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip,
                                         thumb_ip, index_mcp, middle_mcp, ring_mcp, pinky_mcp)
            
            # PRIORITY 2: BRAKE - Fist gesture detection
            braking = self._detect_fist(landmarks, thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip,
                                       thumb_ip, index_mcp, middle_mcp, ring_mcp, pinky_mcp)
            
            # PRIORITY 3: BOOST - Thumb up with other fingers curled
            boost = self._detect_thumbs_up(landmarks, thumb_tip, thumb_ip, index_tip, middle_tip, 
                                          ring_tip, pinky_tip, wrist)
            
            # FIXED: Only calculate steering and throttle if NOT in stop/brake/boost mode
            if stop:
                # STOP gesture overrides everything
                steering = 0.0
                throttle = 0.0
                current_time = time.time()
                if not hasattr(self, '_last_stop_override_log') or current_time - self._last_stop_override_log > 2.0:
                    print(f"🛑 STOP gesture active - overriding steering and throttle")
                    self._last_stop_override_log = current_time
            elif braking:
                # BRAKE gesture allows limited steering but no throttle
                steering = self._calculate_steering(landmarks, wrist, index_mcp) * 0.3  # Reduced steering
                throttle = 0.0
            elif boost:
                # BOOST gesture allows normal steering with max throttle
                steering = self._calculate_steering(landmarks, wrist, index_mcp)
                throttle = 1.0  # Max throttle for boost
            else:
                # Normal driving mode - calculate steering and throttle
                steering = self._calculate_steering(landmarks, wrist, index_mcp)
                throttle = self._calculate_throttle(landmarks, frame_height, hand_center_y)
            
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
        """Calculate throttle based on hand height with continuous acceleration/deceleration"""
        try:
            # Get current time for delta calculation
            current_time = time.time()
            dt = min(0.1, current_time - self.last_throttle_time)
            self.last_throttle_time = current_time
            
            # Use the passed hand_center_y parameter properly
            normalized_height = hand_center_y
            
            # Determine if hand is in upper or lower half of the frame
            if normalized_height < 0.5:  # Upper half - Accelerate
                accel_factor = 1.0 - normalized_height * 2
                throttle_change = self.acceleration_rate * dt * (0.5 + accel_factor)
                self.current_throttle += throttle_change
            else:  # Lower half - Decelerate
                decel_factor = (normalized_height - 0.5) * 2
                throttle_change = self.deceleration_rate * dt * (0.5 + decel_factor)
                self.current_throttle -= throttle_change
            
            # Clamp throttle value with minimum 0.2 and maximum 1.0
            self.current_throttle = max(0.2, min(1.0, self.current_throttle))
            
            return self.current_throttle
            
        except Exception as e:
            print(f"⚠️ Error calculating throttle: {e}")
            return 0.5
    
    def _detect_fist(self, landmarks, thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip,
                     thumb_ip, index_mcp, middle_mcp, ring_mcp, pinky_mcp):
        """Detect fist gesture for braking"""
        try:
            finger_curl_threshold = 0.07
            finger_distances = [
                self._distance(index_tip, index_mcp),
                self._distance(middle_tip, middle_mcp),
                self._distance(ring_tip, ring_mcp),
                self._distance(pinky_tip, pinky_mcp)
            ]
            
            fingers_curled = all(dist < finger_curl_threshold for dist in finger_distances)
            return fingers_curled
            
        except Exception as e:
            print(f"⚠️ Error in fist detection: {e}")
            return False
    
    def _detect_thumbs_up(self, landmarks, thumb_tip, thumb_ip, index_tip, middle_tip, 
                          ring_tip, pinky_tip, wrist):
        """Detect thumbs-up gesture for boost"""
        try:
            thumb_extended_up = thumb_tip[1] < thumb_ip[1]
            
            index_curled = index_tip[1] > landmarks[6][1]
            middle_curled = middle_tip[1] > landmarks[10][1]
            ring_curled = ring_tip[1] > landmarks[14][1]
            pinky_curled = pinky_tip[1] > landmarks[18][1]
            
            all_fingers_curled = index_curled and middle_curled and ring_curled and pinky_curled
            
            return thumb_extended_up and all_fingers_curled
            
        except Exception as e:
            print(f"⚠️ Error in thumbs up detection: {e}")
            return False
    
    def _detect_open_palm(self, landmarks, thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip,
                      thumb_ip, index_mcp, middle_mcp, ring_mcp, pinky_mcp):
        """Detect open palm gesture for stop"""
        try:
            index_pip = landmarks[6]
            middle_pip = landmarks[10]
            ring_pip = landmarks[14]
            pinky_pip = landmarks[18]
            
            index_extended = (index_tip[1] < index_pip[1] - 0.01) and (self._distance(index_tip, index_mcp) > 0.06)
            middle_extended = (middle_tip[1] < middle_pip[1] - 0.01) and (self._distance(middle_tip, middle_mcp) > 0.06)
            ring_extended = (ring_tip[1] < ring_pip[1] - 0.01) and (self._distance(ring_tip, ring_mcp) > 0.06)
            pinky_extended = (pinky_tip[1] < pinky_pip[1] - 0.01) and (self._distance(pinky_tip, pinky_mcp) > 0.06)
            
            wrist = landmarks[0]
            thumb_distance_from_wrist = self._distance(thumb_tip, wrist)
            thumb_extended = thumb_distance_from_wrist > 0.10
            
            total_extended_fingers = sum([
                thumb_extended,
                index_extended, 
                middle_extended,
                ring_extended,
                pinky_extended
            ])
            
            finger_spread_check = (
                self._distance(index_tip, middle_tip) > 0.025 or
                self._distance(middle_tip, ring_tip) > 0.025 or
                self._distance(ring_tip, pinky_tip) > 0.025
            )
            
            palm_center = landmarks[0]
            fingers_away_from_palm = all([
                self._distance(index_tip, palm_center) > 0.12,
                self._distance(middle_tip, palm_center) > 0.12,
                self._distance(ring_tip, palm_center) > 0.12,
                self._distance(pinky_tip, palm_center) > 0.12
            ])
            
            open_palm_detected = (total_extended_fingers >= 4) and finger_spread_check and fingers_away_from_palm
            
            return open_palm_detected
            
        except Exception as e:
            print(f"⚠️ Error in open palm detection: {e}")
            return False
    
    def _calculate_confidence(self, landmarks):
        """Calculate confidence score based on landmark stability"""
        try:
            if len(self.gesture_history) < 3:
                return 0.5
            
            recent_landmarks = [h.get('landmarks') for h in self.gesture_history[-3:] if h.get('landmarks')]
            if not recent_landmarks:
                return 0.5
                
            movement_variance = 0
            for i, landmark in enumerate(landmarks[:5]):
                if i < len(recent_landmarks[0]):
                    recent_positions = [rl[i] for rl in recent_landmarks if i < len(rl)]
                    if recent_positions:
                        x_var = sum((pos[0] - landmark[0])**2 for pos in recent_positions) / len(recent_positions)
                        y_var = sum((pos[1] - landmark[1])**2 for pos in recent_positions) / len(recent_positions)
                        movement_variance += x_var + y_var
            
            confidence = max(0.1, min(1.0, 1.0 - movement_variance * 10))
            return confidence
            
        except Exception:
            return 0.5
    
    def _apply_smoothing(self, raw_gesture):
        """Apply smoothing to reduce gesture jitter"""
        gesture_with_landmarks = raw_gesture.copy()
        self.gesture_history.append(gesture_with_landmarks)
        
        if len(self.gesture_history) > self.max_history:
            self.gesture_history.pop(0)
        
        if len(self.gesture_history) < 3:
            return raw_gesture
        
        smoothed = {}
        
        if raw_gesture.get('stop', False):
            smoothed['steering'] = 0.0
            smoothed['throttle'] = 0.0
        else:
            for key in ['steering', 'throttle']:
                if key in raw_gesture:
                    recent_values = [h[key] for h in self.gesture_history[-5:] if key in h and not h.get('stop', False)]
                    if recent_values:
                        weights = [0.1, 0.15, 0.2, 0.25, 0.3][-len(recent_values):]
                        smoothed[key] = sum(v * w for v, w in zip(recent_values, weights)) / sum(weights)
                    else:
                        smoothed[key] = raw_gesture[key]
                else:
                    smoothed[key] = raw_gesture.get(key, 0.0)
        
        for key in ['braking', 'boost', 'stop']:
            if key in raw_gesture:
                recent_values = [h[key] for h in self.gesture_history[-3:] if key in h]
                if recent_values:
                    if key == 'stop':
                        smoothed[key] = raw_gesture[key]
                    else:
                        smoothed[key] = sum(recent_values) > len(recent_values) / 2
                else:
                    smoothed[key] = raw_gesture[key]
            else:
                smoothed[key] = raw_gesture.get(key, False)
        
        if not smoothed.get('stop', False) and len(self.gesture_history) >= 3:
            recent_steering = [h.get('steering', 0) for h in self.gesture_history[-3:] if not h.get('stop', False)]
            if recent_steering:
                steering_stability = max(recent_steering) - min(recent_steering)
                if steering_stability > self.stability_threshold:
                    smoothed['steering'] *= 0.7
        
        for key in ['confidence', 'detection_fps', 'landmarks']:
            smoothed[key] = raw_gesture.get(key, 0)
        
        return smoothed
    
    def _distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
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
    
    def _determine_gesture_name(self, gesture):
        """Determine the primary gesture name based on gesture data"""
        try:
            if gesture.get('stop', False):
                return 'Open Palm Stop'
            elif gesture.get('braking', False):
                return 'Fist Brake'
            elif gesture.get('boost', False):
                return 'Thumbs Up Boost'
            else:
                steering = gesture.get('steering', 0.0)
                if abs(steering) > 0.3:
                    if steering > 0.3:
                        return 'Turn Right'
                    else:
                        return 'Turn Left'
                else:
                    return 'Forward'
        except Exception as e:
            print(f"⚠️ Error determining gesture name: {e}")
            return 'Unknown'

class SoundManager:
    """Stub class for audio management"""
    def __init__(self):
        self.muted = False
    
    def toggle_mute(self):
        """Toggle the mute state"""
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
    """Draw text on the specified screen"""
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
        """Check collision between the car and an obstacle rectangle"""
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
            
            # Draw car body
            pygame.draw.rect(
                car_surface,
                BLUE,
                (0, 0, self.width, self.height),
                0,
                10
            )
            
            # Draw windshield
            windshield_width = self.width * 0.7
            windshield_height = self.height * 0.3
            pygame.draw.rect(
                car_surface,
                (150, 220, 255),
                (
                    (self.width - windshield_width) / 2,
                    self.height * 0.15,
                    windshield_width,
                    windshield_height
                ),
                0,
                5
            )
            
            # Draw headlights
            light_size = self.width // 5
            pygame.draw.circle(
                car_surface,
                (255, 255, 200),
                (self.width // 4, light_size),
                light_size // 2
            )
            pygame.draw.circle(
                car_surface,
                (255, 255, 200),
                (self.width - self.width // 4, light_size),
                light_size // 2
            )
            
            # Draw brake lights if braking
            if self.braking:
                pygame.draw.circle(
                    car_surface,
                    (255, 0, 0),
                    (self.width // 4, self.height - light_size),
                    light_size // 2
                )
                pygame.draw.circle(
                    car_surface,
                    (255, 0, 0),
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
            # Fallback to simple rectangle
            car_rect = pygame.Rect(
                self.x - self.width // 2,
                self.y - self.height // 2,
                self.width,
                self.height
            )
            pygame.draw.rect(screen, BLUE, car_rect)

class Obstacle:
    """Obstacle class for road hazards - FIXED: Turtles with absolute positioning"""
    def __init__(self, x, y, speed=200, obstacle_type="turtle"):
        self.x = x  # This will be the absolute world position
        self.y = y
        self.world_x = x  # ALWAYS initialize world_x for ALL obstacle types
        self.width = 30
        self.height = 40
        self.speed = speed
        self.hit = False
        self.obstacle_type = obstacle_type
        
        # Only turtle type supported now
        if obstacle_type == "turtle":
            self.color = (255, 165, 0)  # Orange color for turtles
            self.width = 50
            self.height = 35
        else:
            # Fallback to turtle if somehow cone is requested
            self.obstacle_type = "turtle"
            self.color = (255, 165, 0)
            self.width = 50
            self.height = 35
            print(f"⚠️ Cone type not supported - converted to turtle")
        
        self.rect = pygame.Rect(x - self.width // 2, y - self.height // 2, 
                               self.width, self.height)
    
    def update(self, dt):
        """Update obstacle position - ONLY move Y, keep X at world position"""
        self.y += self.speed * dt
        # IMPORTANT: Don't update self.x - it stays at the absolute world position
        # Only update rect for collision detection
        self.rect.centerx = self.x  # Keep at absolute world position
        self.rect.centery = self.y
    
    def _draw_turtle(self, screen, color):
        """Draw a turtle obstacle"""
        # Draw turtle shell (main body)
        shell_rect = pygame.Rect(
            self.x - self.width // 2,
            self.y - self.height // 2,
            self.width,
            self.height
        )
        pygame.draw.ellipse(screen, color, shell_rect)
        
        # Draw shell pattern
        pattern_color = (200, 120, 0)
        for i in range(3):
            for j in range(2):
                pattern_x = self.x - self.width // 3 + i * (self.width // 4)
                pattern_y = self.y - self.height // 3 + j * (self.height // 3)
                pygame.draw.circle(screen, pattern_color, (pattern_x, pattern_y), 4)
        
        # Draw head
        head_x = self.x
        head_y = self.y - self.height // 2 - 8
        pygame.draw.circle(screen, (150, 100, 0), (head_x, head_y), 8)
        
        # Draw eyes
        pygame.draw.circle(screen, BLACK, (head_x - 3, head_y - 2), 2)
        pygame.draw.circle(screen, BLACK, (head_x + 3, head_y - 2), 2)
        
        # Draw legs
        leg_color = (120, 80, 0)
        # Front legs
        pygame.draw.circle(screen, leg_color, (self.x - self.width // 3, self.y + self.height // 4), 6)
        pygame.draw.circle(screen, leg_color, (self.x + self.width // 3, self.y + self.height // 4), 6)
        # Back legs
        pygame.draw.circle(screen, leg_color, (self.x - self.width // 4, self.y + self.height // 2 + 5), 5)
        pygame.draw.circle(screen, leg_color, (self.x + self.width // 4, self.y + self.height // 2 + 5), 5)

    def draw(self, screen):
        """Draw obstacle - ONLY turtles, no cones"""
        color = RED if self.hit else self.color
        
        # Always draw turtle
        self._draw_turtle(screen, color)

class Game:
    """Main game class with steering-based background movement"""
    
    def __init__(self, mode="normal", hand_detector=None, show_tutorial=True, config=None):
        """Initialize the game"""
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
        self.total_dash_cycle = 70
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
        
        # Add game state for menu system
        self._show_main_menu = True
        self._show_mode_selection = False
        self._selected_mode = None
        
        # Create game mode menu buttons
        self._create_mode_selection_buttons()
        
        print(f"✅ Game initialized in {mode} mode")
        if self._hand_detector and self._hand_detector.enabled:
            print("✅ Hand detection enabled")
        else:
            print("⚠️ Hand detection disabled - using keyboard controls")
    
    def _init_camera(self, camera_index=0, width=640, height=480):
        """Initialize camera with robust error handling"""
        try:
            if camera_index is None:
                camera_index = 0
            
            backends = [
                cv2.CAP_DSHOW,
                cv2.CAP_MSMF,
                cv2.CAP_ANY
            ]
            
            for backend in backends:
                try:
                    print(f"🔍 Trying camera {camera_index} with backend {backend}")
                    camera = cv2.VideoCapture(camera_index, backend)
                    
                    if not camera.isOpened():
                        camera.release()
                        continue
                    
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    camera.set(cv2.CAP_PROP_FPS, 30)
                    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
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
            
            print(f"❌ Failed to open any camera")
            return None
            
        except Exception as e:
            print(f"❌ Error initializing camera: {e}")
            return None
    
    def read_frame(self):
        """Read frame from camera with robust error handling"""
        if self._camera is None:
            return False, None
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                ret, frame = self._camera.read()
                
                if ret and frame is not None:
                    return True, frame
                else:
                    if attempt == 0:
                        print(f"⚠️ Camera read failed, attempt {attempt + 1}/{max_retries}")
                    
                    if attempt < max_retries - 1:
                        self._restart_camera()
                        
            except Exception as e:
                print(f"⚠️ Error reading frame (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    self._restart_camera()
        
        return False, None
    
    def process_camera_input(self):
        """Process camera input for gesture detection with enhanced error handling"""
        if self._camera is None or self._hand_detector is None or not self._hand_detector.enabled:
            return None
        
        try:
            success, self._frame = self.read_frame()
            
            if not success or self._frame is None:
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
            
            if not hasattr(self, '_last_camera_restart') or time.time() - self._last_camera_restart > 5.0:
                self._last_camera_restart = time.time()
                self._restart_camera()
            
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

    def update_obstacles(self, current_time, delta_time):
        """Update obstacles - FIXED: Only spawn turtles in easy mode"""
        if self.mode == "practice":
            return
        
        if current_time >= self._next_obstacle_time:
            # FIXED: Only spawn obstacles in easy mode
            if self.mode == "easy":
                road_center = self.screen_width // 2
                road_width = 300
                # FIXED: Generate ABSOLUTE world position, not relative to current offset
                obstacle_world_x = random.randint(
                    road_center - road_width // 2 + 50,
                    road_center + road_width // 2 - 50
                )
                
                obstacle_type = "turtle"
                road_speed = abs(self._car.speed) * 400
                obstacle_speed = road_speed
                print(f"🐢 Spawning turtle at ABSOLUTE world position x={obstacle_world_x}, speed={obstacle_speed:.1f}")
                
                # Store ABSOLUTE world position
                obstacle = Obstacle(obstacle_world_x, -50, obstacle_speed, obstacle_type)
                # world_x is already set in constructor now
                self._obstacles.append(obstacle)
                
                next_spawn_delay = random.uniform(3.0, 6.0)
                print(f"🐢 Next turtle in {next_spawn_delay:.1f} seconds")
                self._next_obstacle_time = current_time + next_spawn_delay
            else:
                # For all other modes, don't spawn obstacles
                print(f"🚫 No obstacles in {self.mode} mode")
                self._next_obstacle_time = current_time + 10.0  # Check again in 10 seconds
        
        # Update obstacles
        for obstacle in self._obstacles[:]:
            if obstacle.obstacle_type == "turtle":
                # Keep original speed but DON'T change the world position
                current_road_speed = abs(self._car.speed) * 400
                obstacle.speed = current_road_speed
                # IMPORTANT: Don't update obstacle.x - keep it at original world position
            
            obstacle.update(delta_time)
            
            if obstacle.y > self.screen_height + 50:
                self._obstacles.remove(obstacle)
                score_bonus = SCORE_PER_OBSTACLE * self.settings["score_multiplier"]
                if obstacle.obstacle_type == "turtle":
                    score_bonus *= 0.5
                    print(f"🐢 Turtle avoided! +{score_bonus:.0f} points")
                self._score += score_bonus

    def draw(self):
        """Draw the game screen with moving background - FIXED turtle positions"""
        if self._show_mode_selection:
            self._draw_mode_selection_menu()
        elif self._show_main_menu:
            self._draw_main_menu()
        else:
            self.screen.fill((50, 50, 50))
            
            if self._moving_road:
                try:
                    self._moving_road.set_steering_offset(self._road_offset_x)
                    self._moving_road.draw(self.screen)
                except Exception as e:
                    print(f"⚠️ Error drawing moving road: {e}")
                    self.draw_built_in_road()
            else:
                self.draw_built_in_road()
            
            # FIXED: Calculate visual offset for turtles only
            steering_visual_offset = int(self._road_offset_x) if hasattr(self, '_road_offset_x') else 0
            
            for obstacle in self._obstacles:
                if obstacle.obstacle_type == "turtle":
                    # FIXED: Calculate screen position from ABSOLUTE world position
                    # Turtles appear to move opposite to steering to stay on road
                    obstacle_screen_x = getattr(obstacle, 'world_x', obstacle.x) - steering_visual_offset
                    
                    # Temporarily set screen position for drawing
                    original_x = obstacle.x
                    obstacle.x = obstacle_screen_x
                    obstacle.draw(self.screen)
                    obstacle.x = original_x  # Restore original position
                else:
                    obstacle.draw(self.screen)
            
            self._car.draw(self.screen)
            
            self._draw_ui()
            self.draw_camera_feed()
            
            if self._paused:
                self._draw_pause()
            
            if self._game_over or self._game_completed:
                self._draw_game_over()
                
            if self._debug_mode:
                self._draw_debug()
            
            if self._show_help:
                self.draw_help()
        
        pygame.display.flip()

    def _check_collisions(self):
        """Check for collisions between car and obstacles - FIXED collision detection"""
        for obstacle in self._obstacles[:]:
            if obstacle.obstacle_type == "turtle":
                # FIXED: Use ABSOLUTE world position for collision detection
                steering_visual_offset = int(self._road_offset_x) if hasattr(self, '_road_offset_x') else 0
                turtle_screen_x = getattr(obstacle, 'world_x', obstacle.x) - steering_visual_offset
                
                turtle_rect = pygame.Rect(
                    turtle_screen_x - obstacle.width // 2,
                    obstacle.y - obstacle.height // 2,
                    obstacle.width,
                    obstacle.height
                )
                if self._car.check_collision(turtle_rect):
                    self._obstacles.remove(obstacle)
                    print("💥 Turtle collision detected! Obstacle removed.")
                    self._score = max(0, self._score - 50)
                    break
            else:
                if self._car.check_collision(obstacle.rect):
                    self._obstacles.remove(obstacle)
                    print("💥 Collision detected! Obstacle removed.")
                    self._score = max(0, self._score - 50)
                    break

    def handle_events(self):
        """Handle all events for the game - FIXED to prevent crashes"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self._show_mode_selection:
                        self._back_to_main_menu()
                    else:
                        self.running = False
                elif event.key == pygame.K_p and not self._show_main_menu and not self._show_mode_selection:
                    self.toggle_pause()
                elif event.key == pygame.K_m and not self._show_main_menu and not self._show_mode_selection:
                    self.toggle_mute()
                elif event.key == pygame.K_h and not self._show_main_menu and not self._show_mode_selection:
                    self._show_help = not self._show_help
                elif event.key == pygame.K_d and not self._show_main_menu and not self._show_mode_selection:
                    self._debug_mode = not self._debug_mode
                    print(f"Debug mode: {'ON' if self._debug_mode else 'OFF'}")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if self._show_main_menu and hasattr(self, 'start_button_rect'):
                        if self.start_button_rect.collidepoint(event.pos):
                            self._show_start_game_menu()
                    elif self._show_mode_selection:
                        for button in getattr(self, 'mode_buttons', []):
                            button.handle_event(event)
                    elif not self._show_main_menu and not self._show_mode_selection:
                        if hasattr(self, 'pause_button') and self.pause_button.rect.collidepoint(event.pos):
                            self.toggle_pause()
                        elif hasattr(self, 'mute_button') and self.mute_button.rect.collidepoint(event.pos):
                            self.toggle_mute()

    def _draw_debug(self):
        """Draw debug information - FIXED to prevent crashes"""
        try:
            debug_info = [
                f"FPS: {self.clock.get_fps():.1f}",
                f"Car Position: ({int(self._car.x)}, {int(self._car.y)})",
                f"Car Speed: {self._car.speed:.2f}",
                f"Road Offset X: {getattr(self, '_road_offset_x', 0):.1f}",
                f"Road Offset Y: {getattr(self, 'road_offset', 0):.1f}",
                f"Obstacles: {len(self._obstacles)}",
                f"Mode: {self.mode}",
                f"Time Left: {getattr(self, 'time_left', 0):.1f}"
            ]
            
            y_offset = self.screen_height - len(debug_info) * 25 - 20
            for info in debug_info:
                debug_text = self._font.render(info, True, WHITE)
                self.screen.blit(debug_text, (10, y_offset))
                y_offset += 25
        except Exception as e:
            print(f"⚠️ Error in debug display: {e}")

    def _draw_main_menu(self):
        """Draw main menu"""
        self.screen.fill((30, 30, 50))
        
        title_text = self._title_font.render("Hand Gesture Car Control", True, WHITE)
        title_rect = title_text.get_rect(center=(self.screen_width // 2, 150))
        self.screen.blit(title_text, title_rect)
        
        # Start button
        self.start_button_rect = pygame.Rect(
            self.screen_width // 2 - 100,
            self.screen_height // 2,
            200,
            50
        )
        pygame.draw.rect(self.screen, PRIMARY, self.start_button_rect)
        
        start_text = self._font.render("START GAME", True, WHITE)
        start_text_rect = start_text.get_rect(center=self.start_button_rect.center)
        self.screen.blit(start_text, start_text_rect)

    def _draw_mode_selection_menu(self):
        """Draw mode selection menu"""
        self.screen.fill((30, 30, 50))
        
        title_text = self._title_font.render("Select Game Mode", True, WHITE)
        title_rect = title_text.get_rect(center=(self.screen_width // 2, 100))
        self.screen.blit(title_text, title_rect)
        
        # Draw mode buttons (if they exist)
        if hasattr(self, 'mode_buttons'):
            for button in self.mode_buttons:
                button.draw(self.screen)

    def _create_mode_selection_buttons(self):
        """Create mode selection buttons"""
        self.mode_buttons = []
        modes = ["practice", "easy", "normal", "hard", "time_trial"]
        
        y_start = 200
        button_height = 60
        button_spacing = 80
        
        for i, mode in enumerate(modes):
            y = y_start + i * button_spacing
            button = Button(
                self.screen_width // 2 - 100,
                y,
                200,
                button_height,
                mode.upper(),
                PRIMARY,
                WHITE,
                lambda m=mode: self._start_game_with_mode(m)
            )
            self.mode_buttons.append(button)

    def _show_start_game_menu(self):
        """Show start game menu"""
        self._show_main_menu = False
        self._show_mode_selection = True

    def _back_to_main_menu(self):
        """Go back to main menu"""
        self._show_main_menu = True
        self._show_mode_selection = False

    def _start_game_with_mode(self, mode):
        """Start game with selected mode"""
        self.mode = mode
        self.settings = GAME_MODES.get(mode, GAME_MODES["normal"])
        self._show_main_menu = False
        self._show_mode_selection = False
        self._start_time = time.time()

    def run(self):
        """Main game loop"""
        self.running = True
        last_time = time.time()
        
        print(f"✅ Starting game in {self.mode} mode")
        
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
            print(f"❌ Error in game loop: {e}")
            raise
        finally:
            self.cleanup()

    def toggle_pause(self):
        """Toggle pause state"""
        self._paused = not self._paused
        print(f"Game {'paused' if self._paused else 'resumed'}")

    def toggle_mute(self):
        """Toggle mute state"""
        muted = self._sound_manager.toggle_mute()
        print(f"Sound {'muted' if muted else 'unmuted'}")
        return muted

    def handle_input(self):
        """Handle keyboard input"""
        return pygame.key.get_pressed()

    def _restart_camera(self):
        """Restart the camera connection"""
        try:
            if self._camera:
                self._camera.release()
            
            time.sleep(0.1)
            
            self._camera = self._init_camera(0, CAMERA_WIDTH, CAMERA_HEIGHT)
            
            if self._camera:
                print("🔄 Camera restarted successfully")
            else:
                print("❌ Camera restart failed")
                
        except Exception as e:
            print(f"❌ Error restarting camera: {e}")
            self._camera = None

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

    def draw_built_in_road(self):
        """Draw built-in road animation with steering compensation - FIXED"""
        # Fill with grass color
        self.screen.fill((0, 100, 0))
        
        # FIXED: Calculate road position with steering offset compensation
        road_width = 300
        steering_offset = int(self._road_offset_x) if hasattr(self, '_road_offset_x') else 0
        road_x = (self.screen_width // 2 - road_width // 2) - steering_offset
        
        # Draw road surface
        pygame.draw.rect(self.screen, (80, 80, 80), (road_x, 0, road_width, self.screen_height))
        
        # FIXED: Draw moving center line - ensure it moves DOWN consistently
        line_width = 6
        dash_length = 50  # Longer dashes for better visibility
        gap_length = 30   # Longer gaps
        dash_spacing = dash_length + gap_length
        
        # CRITICAL FIX: Make sure offset calculation moves dashes downward
        offset = int(self.road_offset) % dash_spacing
        center_x = self.screen_width // 2 - steering_offset
        
        # Draw dashes from top to bottom
        y = -offset  # Start above screen
        while y < self.screen_height + dash_spacing:
            if y + dash_length > 0 and y < self.screen_height:  # Only draw visible dashes
                dash_start = max(0, y)
                dash_end = min(self.screen_height, y + dash_length)
                
                # Draw the dash
                pygame.draw.rect(self.screen, WHITE,
                               (center_x - line_width // 2, dash_start,
                                line_width, dash_end - dash_start))
            
            y += dash_spacing  # Move to next dash position
        
        # Add road edges for better visibility
        pygame.draw.line(self.screen, WHITE, (road_x, 0), (road_x, self.screen_height), 3)
        pygame.draw.line(self.screen, WHITE, (road_x + road_width, 0), (road_x + road_width, self.screen_height), 3)

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

    def cleanup(self):
        """Clean up game resources"""
        try:
            if self._camera:
                print("🧹 Cleaning up camera...")
                self._camera.release()
                self._camera = None
                
            if self._hand_detector:
                print("🧹 Cleaning up hand detector...")
                self._hand_detector.release()
                
            self._sound_manager.cleanup()
            
            cv2.destroyAllWindows()
            time.sleep(0.1)
            
            pygame.quit()
            print("✅ Cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")
            try:
                pygame.quit()
            except:
                pass

    def update(self, dt):
        """Update game state with enhanced road movement - FIXED ROAD ANIMATION"""
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
        gestures = self.process_camera_input()
        if gestures:
            self._last_gesture_result = gestures
        
        keys = self.handle_input()
        
        # FIXED: Only process controls if NOT in menu mode
        if not self._show_main_menu and not self._show_mode_selection:
            # Update car with gesture controls (priority) or keyboard fallback
            if gestures and gestures.get('confidence', 0) > 0.1:
                # Use gesture controls
                self._car.update(gestures, dt)
            else:
                # Fallback to keyboard controls
                keyboard_controls = self._process_keyboard_input(keys)
                self._car.update(keyboard_controls, dt)
            
            # FIXED: Update road animation properly - move downward consistently
            road_speed = 200  # Fixed constant speed for visible movement
            if self._car.speed > 0.1:  # Only move road when car is moving
                road_speed = abs(self._car.speed) * 400  # Scale with car speed
            
            self.road_offset += road_speed * dt  # This makes center line move DOWN
            
            # FIXED: Enhanced horizontal road offset for stronger steering effect
            if hasattr(self._car, 'steering'):
                steering_intensity = abs(self._car.steering) * 2.0
                steering_speed = self._car.steering * steering_intensity * 200 * dt
                self._road_offset_x += steering_speed
            
            # Update MovingRoad if available
            if self._moving_road:
                try:
                    self._moving_road.update(self._car.rotation, self._car.speed, dt)
                except Exception as e:
                    print(f"⚠️ Error updating MovingRoad: {e}")
            
            # Update obstacles
            self.update_obstacles(current_time, dt)
            
            # Check collisions
            self._check_collisions()
            
            # Update score based on distance traveled
            if self._last_position is not None:
                distance = abs(self._car.speed) * dt * 100
                self._distance_traveled += distance
                self._score += distance * 0.1  # Points for distance
            
            self._last_position = (self._car.x, self._car.y)

    def _process_keyboard_input(self, keys):
        """Process keyboard input for fallback controls"""
        controls = {
            'steering': 0.0,
            'throttle': 0.5,
            'braking': False,
            'boost': False
        }
        
        # Steering
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            controls['steering'] = -1.0
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            controls['steering'] = 1.0
        
        # Throttle
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            controls['throttle'] = 1.0
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            controls['throttle'] = 0.2
        
        # Braking
        if keys[pygame.K_SPACE]:
            controls['braking'] = True
            controls['throttle'] = 0.0
        
        # Boost
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            controls['boost'] = True
        
        return controls

    # Add all the missing UI methods
    def _draw_ui(self):
        """Draw UI elements"""
        # Score
        score_text = self._font.render(f"Score: {int(self._score)}", True, WHITE)
        self.screen.blit(score_text, (10, 10))
        
        # Time remaining
        if self.settings["time_limit"] > 0:
            time_text = self._font.render(f"Time: {int(self.time_left)}", True, WHITE)
            self.screen.blit(time_text, (10, 50))
        
        # Speed
        speed_text = self._font.render(f"Speed: {int(self._car.speed * 100)}%", True, WHITE)
        self.screen.blit(speed_text, (10, 90))

    def _draw_pause(self):
        """Draw pause overlay"""
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        pause_text = self._title_font.render("PAUSED", True, WHITE)
        text_rect = pause_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        self.screen.blit(pause_text, text_rect)

    def _draw_game_over(self):
        """Draw game over screen"""
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        if self._game_completed:
            title_text = self._title_font.render("GAME COMPLETED!", True, SUCCESS)
        else:
            title_text = self._title_font.render("GAME OVER", True, ERROR)
        
        title_rect = title_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 50))
        self.screen.blit(title_text, title_rect)
        
        score_text = self._font.render(f"Final Score: {int(self._score)}", True, WHITE)
        score_rect = score_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 20))
        self.screen.blit(score_text, score_rect)

# Add missing run_game function
def run_game(mode="normal", config=None):
    """Run the game with specified mode and configuration"""
    try:
        game = Game(mode=mode, config=config)
        game.run()
    except Exception as e:
        print(f"❌ Error running game: {e}")
        import traceback
        traceback.print_exc()

# Export for import
__all__ = ['Game', 'run_game']
