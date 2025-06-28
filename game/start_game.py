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
        "obstacle_frequency": 0.01,  # Reduced frequency
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
        "time_limit": 60,
        "obstacle_frequency": 0.04,
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
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
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
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            h, w, c = frame.shape
            
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append([cx, cy])
            
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
    """Enhanced gesture recognition for hand control"""
    def __init__(self):
        self.gesture_history = []
        self.max_history = 10
        self.stability_threshold = 0.1
        self.detection_fps = 0
        self.last_detection_time = time.time()
        self.frame_count = 0
        
        self.steering_sensitivity = 1.5
        self.throttle_sensitivity = 1.2
        self.gesture_confidence_threshold = 0.7
        
        self.current_throttle = 0.5
        self.last_throttle_time = time.time()
        
        self.acceleration_rate = 0.8
        self.deceleration_rate = 1.0
        
        print("✅ Enhanced GestureRecognizer initialized")
    
    def recognize_gestures(self, landmarks, frame_height):
        """Enhanced gesture recognition with smoothing"""
        current_time = time.time()
        self.frame_count += 1
        
        if current_time - self.last_detection_time >= 1.0:
            self.detection_fps = self.frame_count
            self.frame_count = 0
            self.last_detection_time = current_time
        
        if not landmarks or len(landmarks) < 21:
            return self._get_default_gestures()
        
        try:
            # Extract key landmarks
            thumb_tip = landmarks[4]
            thumb_ip = landmarks[3]
            thumb_mcp = landmarks[2]
            thumb_base = landmarks[1]
            index_tip = landmarks[8]
            index_mcp = landmarks[5]
            middle_tip = landmarks[12]
            middle_mcp = landmarks[9]
            ring_tip = landmarks[16]
            ring_mcp = landmarks[13]
            pinky_tip = landmarks[20]
            pinky_mcp = landmarks[17]
            wrist = landmarks[0]
            
            hand_center_x = (wrist[0] + middle_mcp[0]) / 2
            hand_center_y = (wrist[1] + middle_mcp[1]) / 2
            
            # Priority 1: STOP - Open palm detection
            stop = self._detect_open_palm(landmarks, thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip,
                                         thumb_ip, index_mcp, middle_mcp, ring_mcp, pinky_mcp)
            
            # Priority 2: BRAKE - Fist gesture detection
            braking = self._detect_fist(landmarks, thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip,
                                       thumb_ip, index_mcp, middle_mcp, ring_mcp, pinky_mcp)
            
            # Priority 3: BOOST - Thumbs up
            boost = self._detect_thumbs_up(landmarks, thumb_tip, thumb_ip, index_tip, middle_tip, 
                                          ring_tip, pinky_tip, wrist)
            
            # Calculate steering and throttle
            if stop:
                steering = 0.0
                throttle = 0.0
            elif braking:
                steering = self._calculate_steering(landmarks, wrist, index_mcp) * 0.3
                throttle = 0.0
            elif boost:
                steering = self._calculate_steering(landmarks, wrist, index_mcp)
                throttle = 1.0
            else:
                steering = self._calculate_steering(landmarks, wrist, index_mcp)
                throttle = self._calculate_throttle(landmarks, frame_height, hand_center_y)
            
            raw_gesture = {
                'steering': steering,
                'throttle': throttle,
                'braking': braking,
                'boost': boost,
                'stop': stop,
                'confidence': self._calculate_confidence(landmarks),
                'detection_fps': self.detection_fps,
                'landmarks': landmarks
            }
            
            smoothed_gesture = self._apply_smoothing(raw_gesture)
            gesture_name = self._determine_gesture_name(smoothed_gesture)
            smoothed_gesture['gesture_name'] = gesture_name
            
            return smoothed_gesture
            
        except Exception as e:
            print(f"⚠️ Error in gesture recognition: {e}")
            return self._get_default_gestures()
    
    def _calculate_steering(self, landmarks, wrist, index_mcp):
        """Calculate steering based on thumb orientation"""
        try:
            thumb_tip = landmarks[4]
            thumb_mcp = landmarks[2]
            
            dx = thumb_tip[0] - thumb_mcp[0]
            dy = thumb_tip[1] - thumb_mcp[1]
            
            angle = math.degrees(math.atan2(dx, -dy))
            
            if angle > 90:
                angle = 180 - angle
            elif angle < -90:
                angle = -180 - angle
            
            max_angle = 75
            steering = angle / max_angle
            steering = max(-1.0, min(1.0, steering))
            
            dead_zone = 0.15
            if abs(steering) < dead_zone:
                steering = 0.0
            else:
                sign = 1 if steering > 0 else -1
                adjusted_value = (abs(steering) - dead_zone) / (1.0 - dead_zone)
                steering = sign * adjusted_value
            
            steering *= self.steering_sensitivity
            steering = max(-1.0, min(1.0, steering))
            
            return steering
            
        except Exception as e:
            print(f"⚠️ Error calculating steering: {e}")
            return 0.0
    
    def _calculate_throttle(self, landmarks, frame_height, hand_center_y):
        """Calculate throttle based on hand height"""
        try:
            current_time = time.time()
            dt = min(0.1, current_time - self.last_throttle_time)
            self.last_throttle_time = current_time
            
            normalized_height = hand_center_y
            
            if normalized_height < 0.5:
                accel_factor = 1.0 - normalized_height * 2
                throttle_change = self.acceleration_rate * dt * (0.5 + accel_factor)
                self.current_throttle += throttle_change
            else:
                decel_factor = (normalized_height - 0.5) * 2
                throttle_change = self.deceleration_rate * dt * (0.5 + decel_factor)
                self.current_throttle -= throttle_change
            
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
        """Determine the primary gesture name"""
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
    """Audio management"""
    def __init__(self):
        self.muted = False
    
    def toggle_mute(self):
        self.muted = not self.muted
        return self.muted
    
    def play(self, sound_name):
        pass
    
    def create_engine(self):
        pass
    
    def cleanup(self):
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


class Car:
    """Car class with camera follow system"""
    def __init__(self, x, y):
        self.world_x = x
        self.world_y = y
        
        self.screen_x = WINDOW_WIDTH // 2
        self.screen_y = WINDOW_HEIGHT - 100
        
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
        
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        
        print("✅ Car initialized with camera follow system")
    
    def update(self, controls, dt, road_bounds=None):
        """Update car state"""
        if controls:
            new_steering = controls.get('steering', 0.0)
            self.throttle = controls.get('throttle', 0.5)
            self.braking = controls.get('braking', False)
            self.boosting = controls.get('boost', False)
            self.steering = new_steering
        
        # Physics
        if self.braking:
            self.speed = max(0.0, self.speed - 2.0 * dt)
        else:
            target_speed = self.throttle
            if self.boosting:
                target_speed = min(1.0, target_speed * 1.5)
            
            self.speed += (target_speed - self.speed) * dt * 2
            self.speed = max(0.0, min(1.0, self.speed))
        
        # Update rotation
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
        
        self.rotation = self.rotation % 360
        if self.rotation > 180:
            self.rotation -= 360
        elif self.rotation < -180:
            self.rotation += 360

        # Movement based on rotation
        if self.speed > 0.05:
            angle_rad = math.radians(self.rotation)
            movement_speed = self.speed * self.max_speed
            self.velocity_x = math.sin(angle_rad) * movement_speed
            self.velocity_y = -math.cos(angle_rad) * movement_speed
            self.world_x += self.velocity_x * dt
            self.world_y += self.velocity_y * dt
        else:
            self.velocity_x = 0.0
            self.velocity_y = 0.0

    def check_collision(self, obstacle_rect):
        """Check collision using screen position"""
        car_rect = pygame.Rect(
            self.screen_x - self.width // 2,
            self.screen_y - self.height // 2,
            self.width,
            self.height
        )
        return car_rect.colliderect(obstacle_rect)

    def draw(self, screen):
        """Draw the car at screen position"""
        try:
            car_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            
            # Draw car body
            pygame.draw.rect(car_surface, BLUE, (0, 0, self.width, self.height), 0, 10)
            
            # Draw windshield
            windshield_width = self.width * 0.7
            windshield_height = self.height * 0.3
            pygame.draw.rect(
                car_surface,
                (150, 220, 255),
                ((self.width - windshield_width) / 2, self.height * 0.15, windshield_width, windshield_height),
                0, 5
            )
            
            # Draw headlights
            light_size = self.width // 5
            pygame.draw.circle(car_surface, (255, 255, 200), (self.width // 4, light_size), light_size // 2)
            pygame.draw.circle(car_surface, (255, 255, 200), (self.width - self.width // 4, light_size), light_size // 2)
            
            # Draw brake lights if braking
            if self.braking:
                pygame.draw.circle(car_surface, RED, (self.width // 4, self.height - light_size), light_size // 2)
                pygame.draw.circle(car_surface, RED, (self.width - self.width // 4, self.height - light_size), light_size // 2)
                
            # Draw boost effect
            if self.boosting:
                flame_points = [
                    (self.width // 2, self.height),
                    (self.width // 2 - self.width // 4, self.height + self.height // 3),
                    (self.width // 2 + self.width // 4, self.height + self.height // 3)
                ]
                pygame.draw.polygon(car_surface, (255, 165, 0), flame_points)
                
            # Rotate the car surface
            rotated_car = pygame.transform.rotate(car_surface, -self.rotation)
            rotated_rect = rotated_car.get_rect(center=(self.screen_x, self.screen_y))
            screen.blit(rotated_car, rotated_rect)
            
        except Exception as e:
            print(f"Error drawing car: {e}")
            car_rect = pygame.Rect(
                self.screen_x - self.width // 2,
                self.screen_y - self.height // 2,
                self.width,
                self.height
            )
            pygame.draw.rect(screen, BLUE, car_rect)


class Obstacle:
    """Obstacle class with world coordinates"""
    def __init__(self, x, y, speed=200, obstacle_type="turtle"):
        self.world_x = x
        self.world_y = y
        self.width = 50
        self.height = 35
        self.speed = speed
        self.hit = False
        self.obstacle_type = obstacle_type
        self.color = (255, 165, 0)  # Orange for turtles
        
        self.screen_x = 0
        self.screen_y = 0
        
        print(f"🐢 Turtle created at world position ({x}, {y})")
    
    def update(self, dt, car_velocity_x=0, car_velocity_y=0):
        """Update obstacle position relative to car movement"""
        self.world_x -= car_velocity_x * dt
        self.world_y -= car_velocity_y * dt
    
    def get_screen_position(self, camera_x, camera_y):
        """Calculate screen position based on camera offset"""
        relative_x = self.world_x - camera_x
        relative_y = self.world_y - camera_y
        self.screen_x = WINDOW_WIDTH // 2 + relative_x
        self.screen_y = WINDOW_HEIGHT - 100 + relative_y
        
        return self.screen_x, self.screen_y
    
    def get_collision_rect(self):
        """Get collision rectangle at screen position"""
        return pygame.Rect(
            self.screen_x - self.width // 2,
            self.screen_y - self.height // 2,
            self.width,
            self.height
        )
    
    def _draw_turtle(self, screen, color):
        """Draw a turtle obstacle"""
        # Shell
        shell_rect = pygame.Rect(
            self.screen_x - self.width // 2,
            self.screen_y - self.height // 2,
            self.width,
            self.height
        )
        pygame.draw.ellipse(screen, color, shell_rect)
        
        # Shell pattern
        pattern_color = (200, 120, 0)
        for i in range(3):
            for j in range(2):
                pattern_x = self.screen_x - self.width // 3 + i * (self.width // 4)
                pattern_y = self.screen_y - self.height // 3 + j * (self.height // 3)
                pygame.draw.circle(screen, pattern_color, (pattern_x, pattern_y), 4)
        
        # Head
        head_x = self.screen_x
        head_y = self.screen_y - self.height // 2 - 8
        pygame.draw.circle(screen, (150, 100, 0), (head_x, head_y), 8)
        
        # Eyes
        pygame.draw.circle(screen, BLACK, (head_x - 3, head_y - 2), 2)
        pygame.draw.circle(screen, BLACK, (head_x + 3, head_y - 2), 2)
        
        # Legs
        leg_color = (120, 80, 0)
        pygame.draw.circle(screen, leg_color, (self.screen_x - self.width // 3, self.screen_y + self.height // 4), 6)
        pygame.draw.circle(screen, leg_color, (self.screen_x + self.width // 3, self.screen_y + self.height // 4), 6)
        pygame.draw.circle(screen, leg_color, (self.screen_x - self.width // 4, self.screen_y + self.height // 2 + 5), 5)
        pygame.draw.circle(screen, leg_color, (self.screen_x + self.width // 4, self.screen_y + self.height // 2 + 5), 5)

    def draw(self, screen, camera_x, camera_y):
        """Draw obstacle at screen position relative to camera"""
        self.get_screen_position(camera_x, camera_y)
        
        if (-100 <= self.screen_x <= WINDOW_WIDTH + 100 and 
            -100 <= self.screen_y <= WINDOW_HEIGHT + 100):
            
            color = RED if self.hit else self.color
            self._draw_turtle(screen, color)


class Game:
    """Main game class with camera follow system"""
    
    def __init__(self, mode="normal", hand_detector=None, show_tutorial=True, config=None):
        """Initialize the game"""
        pygame.init()
        
        self.screen_width = WINDOW_WIDTH
        self.screen_height = WINDOW_HEIGHT
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption(CAPTION)
        
        # Game settings
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
        self._show_help = False
        self.show_camera = True
        
        # Game timing
        self._start_time = 0
        self._elapsed_time = 0
        self._game_duration = self.settings["time_limit"]
        self._time_remaining = self._game_duration
        self.time_left = self._game_duration
        
        # Score and distance
        self._score = 0
        self._distance_traveled = 0
        self._last_position = None
        
        # Car setup
        self._car = Car(0, 0)
        
        # Camera system
        self.camera_x = 0
        self.camera_y = 0
        
        # Obstacles
        self._obstacles = []
        self._next_obstacle_time = 0
        
        # Road animation
        self.road_offset = 0
        self.total_dash_cycle = 70
        
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
        
        # Hand detection
        self._hand_detector = hand_detector or HandDetector()
        self._gesture_recognizer = GestureRecognizer()
        
        # Camera
        self._camera = self._init_camera(0, CAMERA_WIDTH, CAMERA_HEIGHT)
        self._frame = None
        self._last_gesture_result = None
        
        # Sound
        self._sound_manager = SoundManager()
        
        # UI buttons
        self.pause_button = Button(10, self.screen_height - 50, 80, 30, "Pause", PRIMARY, WHITE, self.toggle_pause)
        self.mute_button = Button(100, self.screen_height - 50, 80, 30, "Mute", SECONDARY, WHITE, self.toggle_mute)
        
        # Menu system
        self._show_main_menu = True
        self._show_mode_selection = False
        self._selected_mode = None
        
        self._create_mode_selection_buttons()
        
        print(f"✅ Game initialized in {mode} mode with camera follow")
        if self._hand_detector and self._hand_detector.enabled:
            print("✅ Hand detection enabled")
        else:
            print("⚠️ Hand detection disabled - using keyboard controls")
    
    def _init_camera(self, camera_index=0, width=640, height=480):
        """Initialize camera with error handling"""
        try:
            if camera_index is None:
                camera_index = 0
            
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            
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
        """Read frame from camera"""
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
        """Process camera input for gesture detection"""
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
        """Update obstacles"""
        if self.mode == "practice":
            return

        if current_time >= self._next_obstacle_time:
            if self.mode == "easy":
                road_width = 300
                angle_rad = math.radians(self._car.rotation)
                spawn_distance = 500
                base_x = self._car.world_x + math.sin(angle_rad) * spawn_distance
                base_y = self._car.world_y - math.cos(angle_rad) * spawn_distance
                lateral_offset = random.randint(-road_width // 2 + 50, road_width // 2 - 50)
                perp_angle = angle_rad + math.pi / 2
                obstacle_world_x = base_x + math.cos(perp_angle) * lateral_offset
                obstacle_world_y = base_y + math.sin(perp_angle) * lateral_offset

                obstacle_type = "turtle"
                obstacle_speed = 0
                print(f"🐢 Spawning turtle at world position ({obstacle_world_x:.1f}, {obstacle_world_y:.1f})")
                obstacle = Obstacle(obstacle_world_x, obstacle_world_y, obstacle_speed, obstacle_type)
                self._obstacles.append(obstacle)

                next_spawn_delay = random.uniform(3.0, 6.0)
                print(f"🐢 Next turtle in {next_spawn_delay:.1f} seconds")
                self._next_obstacle_time = current_time + next_spawn_delay
            else:
                print(f"🚫 No obstacles in {self.mode} mode")
                self._next_obstacle_time = current_time + 10.0

        # Update obstacles
        for obstacle in self._obstacles[:]:
            obstacle.update(delta_time, self._car.velocity_x, self._car.velocity_y)
            dx = obstacle.world_x - self._car.world_x
            dy = obstacle.world_y - self._car.world_y
            distance = math.hypot(dx, dy)
            if distance > 1200 or dy > 200:
                self._obstacles.remove(obstacle)
                score_bonus = SCORE_PER_OBSTACLE * self.settings["score_multiplier"]
                if obstacle.obstacle_type == "turtle":
                    score_bonus *= 0.5
                    print(f"🐢 Turtle avoided! +{score_bonus:.0f} points")
                self._score += score_bonus

    def draw_built_in_road(self):
        """Draw built-in road animation"""
        road_width = 300
        road_surface = pygame.Surface((road_width, self.screen_height))
        road_surface.fill((80, 80, 80))

        pygame.draw.line(road_surface, WHITE, (0, 0), (0, self.screen_height), 3)
        pygame.draw.line(road_surface, WHITE, (road_width, 0), (road_width, self.screen_height), 3)

        line_width = 6
        dash_length = 50
        gap_length = 30
        dash_spacing = dash_length + gap_length

        offset = int(self._car.world_y) % dash_spacing
        center_x = road_width // 2
        y = -offset
        while y < self.screen_height + dash_spacing:
            dash_start = max(0, y)
            dash_end = min(self.screen_height, y + dash_length)
            pygame.draw.rect(road_surface, WHITE, (center_x - line_width // 2, dash_start, line_width, dash_end - dash_start))
            y += dash_spacing

        rotated_road = pygame.transform.rotate(road_surface, -self._car.rotation)
        road_rect = rotated_road.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        self.screen.blit(rotated_road, road_rect)

    def update(self, dt):
        """Update game state"""
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
        
        # Update car only if not in menu
        if not self._show_main_menu and not self._show_mode_selection:
            if gestures and gestures.get('confidence', 0) > 0.1:
                self._car.update(gestures, dt)
            else:
                keyboard_controls = self._process_keyboard_input(keys)
                self._car.update(keyboard_controls, dt)
            
            self.update_camera()
            
            if self._moving_road:
                try:
                    self._moving_road.update(self._car.rotation, self._car.speed, dt)
                except Exception as e:
                    print(f"⚠️ Error updating MovingRoad: {e}")
            
            self.update_obstacles(current_time, dt)
            self._check_collisions()
            
            if self._last_position is not None:
                distance = abs(self._car.speed) * dt * 100
                self._distance_traveled += distance
                self._score += distance * 0.1
            self._last_position = (self._car.world_x, self._car.world_y)

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

    def _check_collisions(self):
        """Check for collisions"""
        for obstacle in self._obstacles[:]:
            obstacle.get_screen_position(self.camera_x, self.camera_y)
            obstacle_rect = obstacle.get_collision_rect()
            
            if self._car.check_collision(obstacle_rect):
                self._obstacles.remove(obstacle)
                print("💥 Turtle collision detected! Obstacle removed.")
                self._score = max(0, self._score - 50)
                break

    def _process_keyboard_input(self, keys):
        """Process keyboard input for fallback controls"""
        controls = {
            'steering': 0.0,
            'throttle': 0.5,
            'braking': False,
            'boost': False
        }
        
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            controls['steering'] = -1.0
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            controls['steering'] = 1.0
        
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            controls['throttle'] = 1.0
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            controls['throttle'] = 0.2
        
        if keys[pygame.K_SPACE]:
            controls['braking'] = True
            controls['throttle'] = 0.0
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            controls['boost'] = True
        
        return controls

    def update_camera(self):
        """Camera follows car's world position"""
        self.camera_x = self._car.world_x
        self.camera_y = self._car.world_y

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

    def _start_game_with_mode(self, mode):
        """Start game with selected mode"""
        self.mode = mode
        self.settings = GAME_MODES.get(mode, GAME_MODES["normal"])
        self._show_main_menu = False
        self._show_mode_selection = False
        self._start_time = time.time()

    def run(self):
        """Run the main game loop"""
        self.running = True
        last_time = time.time()
        
        print(f"✅ Starting game in {self.mode} mode with camera follow")
        
        try:
            while self.running:
                current_time = time.time()
                dt = min(0.05, current_time - last_time)
                last_time = current_time
                
                # Process events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            if self._show_mode_selection:
                                self._back_to_main_menu()
                            elif self._show_main_menu:
                                self.running = False
                            elif self._game_over or self._game_completed:
                                self._back_to_main_menu()
                                self._reset_game_state()
                            else:
                                self.toggle_pause()
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
                
                self.update(dt)
                self.draw()
                self.clock.tick(self._target_fps)
        
        except Exception as e:
            print(f"❌ Error in game loop: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()

    def _reset_game_state(self):
        """Reset the game state"""
        self._paused = False
        self._game_over = False
        self._game_completed = False
        self._start_time = 0
        self._elapsed_time = 0
        current_mode_settings = GAME_MODES.get(self.mode, GAME_MODES["normal"])
        self._game_duration = current_mode_settings["time_limit"]
        self._time_remaining = self._game_duration
        self.time_left = self._game_duration
        self._score = 0
        self._distance_traveled = 0
        self._obstacles.clear()
        self._next_obstacle_time = 0
        self._car = Car(0, 0)
        self.camera_x = 0
        self.camera_y = 0
        if self._moving_road and hasattr(self._moving_road, 'reset'):
            self._moving_road.reset()
        print("🔄 Game state reset.")

    def _restart_camera(self):
        """Restart camera connection"""
        try:
            if self._camera:
                self._camera.release()
                time.sleep(0.1)
            
            self._camera = self._init_camera(0, CAMERA_WIDTH, CAMERA_HEIGHT)
            print("🔄 Camera restarted")
        except Exception as e:
            print(f"⚠️ Error restarting camera: {e}")
            self._camera = None

    def _draw_main_menu(self):
        """Draw the main menu screen"""
        self.screen.fill(BLACK)
        
        # Title
        title_text = self._title_font.render("Hand Gesture Car Control", True, WHITE)
        title_rect = title_text.get_rect(center=(self.screen_width // 2, 100))
        self.screen.blit(title_text, title_rect)
        
        # Subtitle
        subtitle_text = self._font.render("Control your car with hand gestures!", True, WHITE)
        subtitle_rect = subtitle_text.get_rect(center=(self.screen_width // 2, 150))
        self.screen.blit(subtitle_text, subtitle_rect)
        
        # Start button
        button_width = 200
        button_height = 60
        button_x = self.screen_width // 2 - button_width // 2
        button_y = 250
        
        self.start_button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
        pygame.draw.rect(self.screen, PRIMARY, self.start_button_rect)
        pygame.draw.rect(self.screen, WHITE, self.start_button_rect, 3)
        
        start_text = self._font.render("START GAME", True, WHITE)
        start_rect = start_text.get_rect(center=self.start_button_rect.center)
        self.screen.blit(start_text, start_rect)
        
        # Instructions
        instructions = [
            "Hand Gestures:",
            "• Open Palm - STOP",
            "• Closed Fist - BRAKE", 
            "• Thumbs Up - BOOST",
            "• Thumb Direction - STEERING",
            "• Hand Height - THROTTLE",
            "",
            "Keyboard Controls:",
            "• Arrow Keys / WASD - Movement",
            "• Space - Brake",
            "• Shift - Boost",
            "• ESC - Menu/Pause",
            "• P - Pause",
            "• M - Mute",
            "• H - Help"
        ]
        
        y_start = 350
        for i, instruction in enumerate(instructions):
            color = ACCENT if instruction.startswith("•") else WHITE
            font = pygame.font.Font(None, 24) if instruction.startswith("•") else pygame.font.Font(None, 28)
            text = font.render(instruction, True, color)
            self.screen.blit(text, (50, y_start + i * 25))

    def _draw_mode_selection_menu(self):
        """Draw the mode selection menu"""
        self.screen.fill(BLACK)
        
        # Title
        title_text = self._title_font.render("Select Game Mode", True, WHITE)
        title_rect = title_text.get_rect(center=(self.screen_width // 2, 100))
        self.screen.blit(title_text, title_rect)
        
        # Mode descriptions
        mode_descriptions = {
            "practice": "No obstacles, no time limit - Perfect for learning",
            "easy": "Few turtles, 4 minutes - Good for beginners", 
            "normal": "Standard difficulty, 3 minutes - Balanced challenge",
            "hard": "More obstacles, 2 minutes - For experts",
            "time_trial": "Maximum challenge, 1 minute - Speed run!"
        }
        
        # Draw mode buttons with descriptions
        for i, button in enumerate(self.mode_buttons):
            button.draw(self.screen)
            
            # Draw description below button
            mode_name = ["practice", "easy", "normal", "hard", "time_trial"][i]
            description = mode_descriptions.get(mode_name, "")
            desc_font = pygame.font.Font(None, 20)
            desc_text = desc_font.render(description, True, WHITE)
            desc_rect = desc_text.get_rect(center=(button.rect.centerx, button.rect.bottom + 15))
            self.screen.blit(desc_text, desc_rect)
        
        # Back instruction
        back_text = self._font.render("Press ESC to go back", True, WHITE)
        back_rect = back_text.get_rect(center=(self.screen_width // 2, self.screen_height - 50))
        self.screen.blit(back_text, back_rect)

    def _show_start_game_menu(self):
        """Show the game mode selection menu"""
        self._show_main_menu = False
        self._show_mode_selection = True

    def _back_to_main_menu(self):
        """Return to main menu"""
        self._show_main_menu = True
        self._show_mode_selection = False

    def _draw_ui(self):
        """Draw the game UI"""
        try:
            # Score
            score_text = f"Score: {int(self._score)}"
            score_surface = self._font.render(score_text, True, WHITE)
            self.screen.blit(score_surface, (10, 10))
            
            # Distance
            distance_text = f"Distance: {int(self._distance_traveled)}m"
            distance_surface = self._font.render(distance_text, True, WHITE)
            self.screen.blit(distance_surface, (10, 50))
            
            # Time remaining
            if self.settings["time_limit"] > 0:
                time_text = f"Time: {int(max(0, self.time_left))}s"
                time_color = RED if self.time_left < 30 else WHITE
                time_surface = self._font.render(time_text, True, time_color)
                self.screen.blit(time_surface, (10, 90))
            
            # Mode indicator
            mode_text = f"Mode: {self.mode.upper()}"
            mode_surface = pygame.font.Font(None, 24).render(mode_text, True, ACCENT)
            self.screen.blit(mode_surface, (10, 130))
            
            # Speed indicator
            speed_percent = int(self._car.speed * 100)
            speed_text = f"Speed: {speed_percent}%"
            speed_surface = pygame.font.Font(None, 24).render(speed_text, True, WHITE)
            self.screen.blit(speed_surface, (10, 155))
            
            # Car status indicators
            status_y = 180
            if self._car.braking:
                brake_text = pygame.font.Font(None, 24).render("BRAKING", True, RED)
                self.screen.blit(brake_text, (10, status_y))
                status_y += 25
            
            if self._car.boosting:
                boost_text = pygame.font.Font(None, 24).render("BOOST!", True, ACCENT)
                self.screen.blit(boost_text, (10, status_y))
                status_y += 25
            
            # Gesture info
            if hasattr(self, '_last_gesture_result') and self._last_gesture_result:
                gesture_name = self._last_gesture_result.get('gesture_name', 'Unknown')
                confidence = self._last_gesture_result.get('confidence', 0)
                
                if gesture_name != 'No Hand Detected' and confidence > 0.1:
                    gesture_color = GREEN if confidence > 0.5 else ACCENT
                    gesture_text = pygame.font.Font(None, 20).render(f"Gesture: {gesture_name}", True, gesture_color)
                    self.screen.blit(gesture_text, (10, self.screen_height - 120))
                    
                    confidence_text = pygame.font.Font(None, 18).render(f"Confidence: {confidence:.1f}", True, WHITE)
                    self.screen.blit(confidence_text, (10, self.screen_height - 100))
            
            # Controls reminder
            controls_text = pygame.font.Font(None, 18).render("ESC: Menu | P: Pause | M: Mute | H: Help", True, WHITE)
            self.screen.blit(controls_text, (10, self.screen_height - 80))
            
            # Draw buttons
            if hasattr(self, 'pause_button'):
                self.pause_button.draw(self.screen)
            if hasattr(self, 'mute_button'):
                self.mute_button.draw(self.screen)
        
        except Exception as e:
            print(f"⚠️ Error drawing UI: {e}")

    def _draw_pause(self):
        """Draw pause overlay"""
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.set_alpha(128)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        pause_text = self._title_font.render("PAUSED", True, WHITE)
        pause_rect = pause_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 50))
        self.screen.blit(pause_text, pause_rect)
        
        instruction_text = self._font.render("Press P or ESC to resume", True, WHITE)
        instruction_rect = instruction_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 20))
        self.screen.blit(instruction_text, instruction_rect)

    def _draw_game_over(self):
        """Draw game over screen"""
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.set_alpha(180)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        if self._game_completed:
            title_text = self._title_font.render("GAME COMPLETED!", True, SUCCESS)
        else:
            title_text = self._title_font.render("GAME OVER", True, ERROR)
        
        title_rect = title_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 100))
        self.screen.blit(title_text, title_rect)
        
        # Final score
        score_text = self._font.render(f"Final Score: {int(self._score)}", True, WHITE)
        score_rect = score_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 50))
        self.screen.blit(score_text, score_rect)
        
        # Distance
        distance_text = self._font.render(f"Distance Traveled: {int(self._distance_traveled)}m", True, WHITE)
        distance_rect = distance_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 20))
        self.screen.blit(distance_text, distance_rect)
        
        # Time survived
        time_text = self._font.render(f"Time Survived: {int(self._elapsed_time)}s", True, WHITE)
        time_rect = time_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 10))
        self.screen.blit(time_text, time_rect)
        
        # Instructions
        instruction_text = self._font.render("Press ESC to return to menu", True, ACCENT)
        instruction_rect = instruction_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 60))
        self.screen.blit(instruction_text, instruction_rect)

    def draw_help(self):
        """Draw help overlay"""
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.set_alpha(200)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        help_title = self._font.render("HELP & CONTROLS", True, WHITE)
        help_rect = help_title.get_rect(center=(self.screen_width // 2, 50))
        self.screen.blit(help_title, help_rect)
        
        help_content = [
            "HAND GESTURES:",
            "• Open Palm (5 fingers) = STOP",
            "• Closed Fist = BRAKE",
            "• Thumbs Up = BOOST",
            "• Thumb Left/Right = STEERING",
            "• Hand Up/Down = THROTTLE",
            "",
            "KEYBOARD CONTROLS:",
            "• Arrow Keys / WASD = Movement",
            "• Space = Brake",
            "• Shift = Boost", 
            "",
            "GAME CONTROLS:",
            "• ESC = Menu/Pause",
            "• P = Pause/Resume",
            "• M = Mute/Unmute",
            "• H = Toggle Help",
            "• D = Debug Mode",
            "",
            "Press H to close help"
        ]
        
        y_start = 100
        for i, line in enumerate(help_content):
            if line.startswith("•"):
                color = ACCENT
                font = pygame.font.Font(None, 24)
            elif line.isupper() and line.endswith(":"):
                color = SUCCESS
                font = pygame.font.Font(None, 28)
            elif line == "":
                continue
            else:
                color = WHITE
                font = pygame.font.Font(None, 24)
            
            text_surface = font.render(line, True, color)
            self.screen.blit(text_surface, (100, y_start + i * 25))

    def cleanup(self):
        """Clean up game resources"""
        try:
            print("🧹 Cleaning up game resources...")
            
            # Release camera
            if hasattr(self, '_camera') and self._camera:
                try:
                    self._camera.release()
                    print("✅ Camera released")
                except Exception as e:
                    print(f"⚠️ Error releasing camera: {e}")
            
            # Release hand detector
            if hasattr(self, '_hand_detector') and self._hand_detector:
                try:
                    self._hand_detector.release()
                    print("✅ Hand detector released")
                except Exception as e:
                    print(f"⚠️ Error releasing hand detector: {e}")
            
            # Clean up sound manager
            if hasattr(self, '_sound_manager') and self._sound_manager:
                try:
                    self._sound_manager.cleanup()
                    print("✅ Sound manager cleaned up")
                except Exception as e:
                    print(f"⚠️ Error cleaning up sound manager: {e}")
            
            # Quit pygame
            try:
                pygame.quit()
                print("✅ Pygame quit successfully")
            except Exception as e:
                print(f"⚠️ Error quitting pygame: {e}")
            
            # Release OpenCV resources
            try:
                cv2.destroyAllWindows()
                print("✅ OpenCV windows destroyed")
            except Exception as e:
                print(f"⚠️ Error destroying OpenCV windows: {e}")
            
            print("✅ Cleanup completed")
            
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")

    def draw(self):
        """Draw the game screen"""
        if self._show_mode_selection:
            self._draw_mode_selection_menu()
        elif self._show_main_menu:
            self._draw_main_menu()
        else:
            self.screen.fill((0, 100, 0))
            
            if self._moving_road:
                try:
                    self._moving_road.draw(self.screen)
                except Exception as e:
                    print(f"⚠️ Error drawing moving road: {e}")
                    self.draw_built_in_road()
            else:
                self.draw_built_in_road()
            
            for obstacle in self._obstacles:
                obstacle.draw(self.screen, self.camera_x, self.camera_y)
            
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

    def draw_camera_feed(self):
        """Draw camera feed in the corner"""
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
                print(f"⚠️ Error drawing camera feed: {e}")

    def _draw_debug(self):
        """Draw debug information"""
        try:
            debug_info = [
                f"FPS: {self.clock.get_fps():.1f}",
                f"Car World: ({int(self._car.world_x)}, {int(self._car.world_y)})",
                f"Car Screen: ({int(self._car.screen_x)}, {int(self._car.screen_y)})",
                f"Camera: ({int(self.camera_x)}, {int(self.camera_y)})",
                f"Car Speed: {self._car.speed:.2f}",
                f"Car Rotation: {self._car.rotation:.1f}°",
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
__all__ = ['Game', 'run_game', 'HandDetector']


if __name__ == "__main__":
    """Main entry point for the game"""
    print("🎮 Starting Hand Gesture Car Control Game")
    run_game("normal")