"""
Enhanced game/start_game.py - WITH STEERING BACKGROUND MOVEMENT
Fixed road direction, proper syntax, and clean structure
"""
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
    """Stub class for hand detection"""
    def __init__(self):
        pass
    
    def find_hands(self, frame):
        return frame, None
    
    def find_positions(self, frame, results):
        return None
    
    def release(self):
        pass

class GestureRecognizer:
    """Stub class for gesture recognition"""
    def __init__(self):
        pass
    
    def recognize_gestures(self, landmarks, frame_height):
        return {
            'steering': 0.0,
            'throttle': 0.5,
            'braking': False,
            'boost': False,
            'gesture_name': 'No detection'
        }

class SoundManager:
    """Stub class for sound management"""
    def __init__(self):
        self.muted = False
    
    def toggle_mute(self):
        self.muted = not self.muted
        return self.muted
    
    def play(self, sound_name):
        pass
    
    def create_engine_sound(self, speed, max_speed):
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

def draw_text(screen, text, font, color, x, y, align="left"):
    """Simple text drawing function"""
    text_surface = font.render(text, True, color)
    
    if align == "center":
        text_rect = text_surface.get_rect(center=(x, y))
        screen.blit(text_surface, text_rect)
    elif align == "right":
        text_rect = text_surface.get_rect()
        text_rect.right = x
        text_rect.y = y
        screen.blit(text_surface, text_rect)
    else:  # left
        screen.blit(text_surface, (x, y))

def create_progress_bar(width, height, progress, color):
    """Create a progress bar surface"""
    bar_surface = pygame.Surface((width, height))
    bar_surface.fill((100, 100, 100))  # Background
    
    fill_width = int(width * progress)
    if fill_width > 0:
        pygame.draw.rect(bar_surface, color, (0, 0, fill_width, height))
    
    return bar_surface

def init_camera(camera_index=None, width=640, height=480):
    """Initialize camera"""
    try:
        if camera_index is None:
            camera_index = 0
        
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            return cap
        else:
            print(f"Could not open camera {camera_index}")
            return None
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return None

def read_frame(camera):
    """Read frame from camera"""
    if camera is None:
        return False, None
    
    try:
        ret, frame = camera.read()
        return ret, frame
    except Exception as e:
        print(f"Error reading frame: {e}")
        return False, None

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
        """Update car state with improved steering response"""
        if controls:
            # קבלת ערכי בקרה עם reset מלא של ההיגוי
            new_steering = controls.get('steering', 0.0)
            self.throttle = controls.get('throttle', 0.5)
            self.braking = controls.get('braking', False)
            self.boosting = controls.get('boost', False)
            
            # עדכון מיידי של ההיגוי (לא מצטבר)
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
        
        # Update rotation based on steering - FIXED VERSION
        if self.speed > 0.05:  # Only turn when moving
            # חישוב שינוי סיבוב על בסיס ההיגוי הנוכחי בלבד
            max_turn_rate = 120  # מקסימום מעלות לשנייה
            turn_amount = self.steering * max_turn_rate * dt
            
            # הפחתת רגישות ההיגוי במהירויות גבוהות
            speed_factor = max(0.3, 1.0 - (self.speed * 0.7))
            turn_amount *= speed_factor
            
            # עדכון הסיבוב
            self.rotation += turn_amount
        else:
            # כשהמכונית עומדת, החזרה הדרגתית למרכז
            if abs(self.rotation) > 1.0:
                center_return = -self.rotation * 0.05  # החזרה איטה למרכז
                self.rotation += center_return * dt * 60
        
        # שמירה על טווח הסיבוב
        self.rotation = self.rotation % 360
        if self.rotation > 180:
            self.rotation -= 360
        elif self.rotation < -180:
            self.rotation += 360

    def draw(self, screen):
        """Draw the car"""
        car_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # Car body
        pygame.draw.rect(car_surface, PRIMARY, (0, 0, self.width, self.height), 0, 8)
        
        # Windshield
        pygame.draw.rect(car_surface, (100, 150, 255), (5, 5, self.width - 10, self.height // 3), 0, 5)
        
        # Headlights
        pygame.draw.circle(car_surface, WHITE, (self.width // 4, 8), 4)
        pygame.draw.circle(car_surface, WHITE, (3 * self.width // 4, 8), 4)
        
        # Brake lights
        if self.braking:
            pygame.draw.circle(car_surface, RED, (self.width // 4, self.height - 8), 6)
            pygame.draw.circle(car_surface, RED, (3 * self.width // 4, self.height - 8), 6)
        else:
            pygame.draw.circle(car_surface, (100, 0, 0), (self.width // 4, self.height - 8), 4)
            pygame.draw.circle(car_surface, (100, 0, 0), (3 * self.width // 4, self.height - 8), 4)
        
        # Boost effect
        if self.boosting:
            flame_points = [
                (self.width // 2, self.height),
                (self.width // 2 - 10, self.height + 20),
                (self.width // 2 + 10, self.height + 20)
            ]
            pygame.draw.polygon(car_surface, (255, 165, 0), flame_points)
        
        # Rotate and draw
        rotated_car = pygame.transform.rotate(car_surface, -self.rotation)
        car_rect = rotated_car.get_rect(center=(self.x, self.y))
        screen.blit(rotated_car, car_rect)
    
    def check_collision(self, obstacle_rect):
        """Check collision with obstacle"""
        car_rect = pygame.Rect(self.x - self.width // 2, self.y - self.height // 2, 
                              self.width, self.height)
        return car_rect.colliderect(obstacle_rect)
    
    def reset(self, x=None, y=None):
        """Reset car to initial state"""
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        self.speed = 0.0
        self.steering = 0.0
        self.throttle = 0.5
        self.braking = False
        self.boosting = False
        self.rotation = 0.0
        self.health = 100

class Obstacle:
    """Obstacle class for road hazards"""
    def __init__(self, x, y, speed=200):
        self.x = x
        self.y = y
        self.width = 30
        self.height = 40
        self.speed = speed
        self.hit = False
        self.color = (255, 140, 0)  # Orange cone
        self.rect = pygame.Rect(x - self.width // 2, y - self.height // 2, 
                               self.width, self.height)
    
    def update(self, dt):
        """Update obstacle position"""
        self.y += self.speed * dt
        self.rect.y = self.y - self.height // 2
    
    def draw(self, screen):
        """Draw obstacle as traffic cone"""
        color = RED if self.hit else self.color
        
        # Draw cone shape
        points = [
            (self.x, self.y - self.height // 2),
            (self.x - self.width // 2, self.y + self.height // 2),
            (self.x + self.width // 2, self.y + self.height // 2)
        ]
        pygame.draw.polygon(screen, color, points)
        
        # White stripe
        pygame.draw.line(screen, WHITE,
                        (self.x - self.width // 4, self.y),
                        (self.x + self.width // 4, self.y), 3)

class Game:
    """Main game class - WITH STEERING BACKGROUND MOVEMENT"""
    
    def __init__(self, mode="normal", screen_width=800, screen_height=600, hand_detector=None):
        """Initialize the game"""
        self.mode = mode
        self.settings = GAME_MODES.get(mode, GAME_MODES["normal"])
        
        # Game state
        self.running = False
        self.paused = False
        self.game_over = False
        
        # Score and time
        self.score = 0
        self.time_left = self.settings["time_limit"]
        self.start_time = 0
        
        # Screen dimensions
        self.screen_width = screen_width or WINDOW_WIDTH
        self.screen_height = screen_height or WINDOW_HEIGHT
        
        # Road animation properties - WITH HORIZONTAL STEERING
        self.road_offset = 0
        self.road_offset_x = 0  # NEW: Horizontal road offset for steering
        self.dash_length = 30
        self.gap_length = 20
        self.total_dash_cycle = self.dash_length + self.gap_length
        
        # Initialize pygame
        if not pygame.get_init():
            pygame.init()
        
        # Screen and display
        self.screen = pygame.display.set_mode((self.screen_width + 320, self.screen_height))
        pygame.display.set_caption(f"{CAPTION} - {mode.capitalize()} Mode - FIXED")
        self.clock = pygame.time.Clock()
        self.target_fps = FPS
        
        # Game objects
        self.car = Car(self.screen_width // 2, self.screen_height - 100)
        self.obstacles = []
        self.next_obstacle_time = 0
        
        # Moving road (if available)
        if MOVING_ROAD_AVAILABLE:
            try:
                self.moving_road = MovingRoadGenerator(self.screen_width, self.screen_height)
                print("✅ MovingRoadGenerator initialized")
            except Exception as e:
                print(f"⚠️ Error initializing MovingRoadGenerator: {e}")
                self.moving_road = None
        else:
            self.moving_road = None
        
        # Camera and gesture detection
        self.camera = None
        self.frame = None
        self.hand_detector = hand_detector
        self.gesture_recognizer = None
        self.init_camera_and_detector()
        
        # Sound system
        self.sound_manager = SoundManager()
        
        # UI elements
        self.create_ui_elements()
        self.camera_surface = None
        
        # Game state flags
        self.debug_mode = True
        self.show_help = True
        self.show_camera = True
        
        # Timer settings
        self.game_duration = 3 * 60  # 3 minutes
        self.elapsed_time = 0
        self.time_remaining = self.game_duration
        self.game_completed = False
        
        # Score tracking
        self.distance_traveled = 0
        self.last_position = None
        
        # Debug settings
        self.debug_input = True
        self.last_debug_time = 0
        
        # Fonts
        self.font = pygame.font.SysFont(None, 24)
        self.title_font = pygame.font.SysFont(None, 36)
        
        logger.info(f"✅ Game initialized in {mode} mode - CLEAN VERSION")
    
    def init_camera_and_detector(self):
        """Initialize camera and hand detection"""
        try:
            camera_index = None
            if 'SELECTED_CAMERA' in os.environ:
                try:
                    camera_index = int(os.environ['SELECTED_CAMERA'])
                except ValueError:
                    pass
            
            self.camera = init_camera(camera_index, CAMERA_WIDTH, CAMERA_HEIGHT)
            
            if self.camera is None:
                logger.error("Failed to initialize camera")
            else:
                if not self.hand_detector:
                    self.hand_detector = HandDetector()
                self.gesture_recognizer = GestureRecognizer()
                logger.info("Camera and hand detector initialized")
                
        except Exception as e:
            logger.error(f"Error initializing camera/detector: {e}")
            self.camera = None
            self.hand_detector = None
            self.gesture_recognizer = None
    
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
    
    def toggle_pause(self):
        """Toggle game pause state"""
        self.paused = not self.paused
        logger.info(f"Game {'paused' if self.paused else 'resumed'}")
    
    def toggle_mute(self):
        """Toggle sound mute state"""
        try:
            is_muted = self.sound_manager.toggle_mute()
            logger.info(f"Sound {'muted' if is_muted else 'unmuted'}")
        except Exception as e:
            logger.error(f"Error toggling mute: {e}")
    
    def handle_input(self):
        """Handle keyboard input with debug output"""
        keys = pygame.key.get_pressed()
        current_time = time.time()
        
        # Debug output (once per second)
        if self.debug_input and current_time - self.last_debug_time > 1.0:
            any_key_pressed = any([
                keys[pygame.K_UP], keys[pygame.K_DOWN], 
                keys[pygame.K_LEFT], keys[pygame.K_RIGHT],
                keys[pygame.K_w], keys[pygame.K_s],
                keys[pygame.K_a], keys[pygame.K_d],
                keys[pygame.K_SPACE]
            ])
            
            if any_key_pressed:
                print(f"🎮 Input: UP={keys[pygame.K_UP]}, DOWN={keys[pygame.K_DOWN]}, LEFT={keys[pygame.K_LEFT]}, RIGHT={keys[pygame.K_RIGHT]}")
                print(f"🚗 Car: speed={self.car.speed:.2f}, steering={self.car.steering:.2f}")
            
            self.last_debug_time = current_time
        
        return keys

    def handle_events(self):
        """Handle pygame events with proper steering reset"""
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
            
            elif event.type == pygame.KEYUP:
                # אין עוד עדכון ישיר של ההיגוי כאן - זה נעשה ב-update()
                pass
            
            # Handle UI buttons
            self.pause_button.handle_event(event)
            self.mute_button.handle_event(event)

    def update(self, delta_time):
        """Update game state with improved steering control"""
        if self.paused or self.game_over:
            return
        
        # Update timing
        current_time = time.time()
        if self.start_time == 0:
            self.start_time = current_time
        
        self.elapsed_time = current_time - self.start_time
        self.time_remaining = max(0, self.game_duration - self.elapsed_time)
        
        # Check time limits
        if self.settings["time_limit"] > 0:
            elapsed = current_time - self.start_time
            self.time_left = max(0, self.settings["time_limit"] - elapsed)
            
            if self.time_left <= 0:
                self.game_over = True
                logger.info("Game over: Time's up")
                return
        
        # Check completion
        if self.elapsed_time >= self.game_duration and not self.game_completed:
            self.game_completed = True
            print(f"⏱️ Game completed! Final score: {int(self.score)}")
        
        # Get input with proper steering reset
        gestures = self.process_camera_input()
        keys = self.handle_input()
        
        # יצירת מילון בקרה חדש בכל פריים
        controls = {
            'steering': 0.0,  # התחלה מאפס
            'throttle': 0.5,
            'braking': False,
            'boost': False
        }
        
        # עדכון על בסיס מחוות (אם קיימות)
        if gestures:
            controls.update(gestures)
        
        # עדכון על בסיס מקלדת - בודק מקשים שנלחצו כרגע
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            controls['steering'] = -1.0  # שמאלה
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            controls['steering'] = 1.0   # ימינה
        # אם אף מקש לא נלחץ, ההיגוי נשאר 0.0
        
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            controls['throttle'] = 1.0
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            controls['throttle'] = 0.0
            controls['braking'] = True
        
        if keys[pygame.K_SPACE]:
            controls['boost'] = True
        
        # עדכון המכונית עם הבקרות החדשות
        self.car.update(controls, delta_time)
        
        # Update road animation - WITH ENHANCED STEERING OFFSET
        road_speed = abs(self.car.speed) * 300
        self.road_offset += road_speed * delta_time
        
        # ENHANCED: Stronger horizontal road offset based on car rotation
        steering_factor = self.car.rotation / 20.0  # More sensitive (20 degrees instead of 45)
        steering_factor = max(-2.5, min(2.5, steering_factor))  # Allow even stronger movement
        horizontal_speed = steering_factor * 180 * self.car.speed  # Increased base speed
        self.road_offset_x += horizontal_speed * delta_time
        
        if self.road_offset >= self.total_dash_cycle:
            self.road_offset -= self.total_dash_cycle
        
        # Update moving road with enhanced steering
        if self.moving_road:
            try:
                self.moving_road.update(self.car.rotation, self.car.speed, delta_time)
            except Exception as e:
                logger.error(f"Error updating moving road: {e}")
                import traceback
                traceback.print_exc()
        
        # Update score
        current_position = (self.car.x, self.car.y)
        if self.last_position:
            distance = math.hypot(current_position[0] - self.last_position[0],
                                current_position[1] - self.last_position[1])
            self.distance_traveled += distance
            self.score = self.distance_traveled / 10
        self.last_position = current_position
        
        # Update obstacles and check collisions
        self.update_obstacles(current_time, delta_time)
        self.check_collisions()
    
    # הוסרו כפילויות של פונקציות
    
    def check_collisions(self):
        """Check collisions between car and obstacles"""
        for obstacle in self.obstacles:
            if not obstacle.hit:
                # גישה בטוחה יותר לתכונה world_x
                world_x = getattr(obstacle, 'world_x', obstacle.x)
                screen_x = world_x - self.road_offset_x
                
                # עדכון המיקום הזמני של המכשול לבדיקת התנגשות
                original_x = obstacle.x
                obstacle.x = screen_x
                
                if self.car.check_collision(obstacle.rect):
                    obstacle.hit = True
                    try:
                        self.sound_manager.play("collision")
                    except Exception as e:
                        logger.error(f"Error playing collision sound: {e}")
                    logger.info("Collision detected")
                    
                    # In hard mode, game over on collision
                    if self.mode == "hard":
                        self.game_over = True
                        logger.info("Game over: Collision in hard mode")
                
                # החזרת המיקום המקורי
                obstacle.x = original_x

    def draw_built_in_road(self):
        """Draw built-in road with steering-based horizontal movement"""
        # Road surface
        road_color = (80, 80, 80)
        road_width = self.screen_width - 100
        
        # Calculate road position with enhanced horizontal offset
        road_x = 50 - int(self.road_offset_x * 1.3)  # Multiply by 1.3 for stronger effect
        
        pygame.draw.rect(self.screen, road_color, (road_x, 0, road_width, self.screen_height))
        
        # Moving center line with enhanced horizontal offset
        center_x = self.screen_width // 2 - int(self.road_offset_x * 1.3)
        line_width = 5
        
        # Draw moving dashed center line
        dash_length = 30
        gap_length = 20
        total_cycle = dash_length + gap_length
        
        # NEGATIVE road_offset to move in same direction as cones
        offset = int(-self.road_offset) % total_cycle
        
        y = -offset
        while y < self.screen_height + total_cycle:
            if y + dash_length > 0 and y < self.screen_height:
                dash_start = max(0, y)
                dash_end = min(self.screen_height, y + dash_length)
                
                # Only draw if line is within screen bounds
                if center_x - line_width // 2 >= 0 and center_x + line_width // 2 <= self.screen_width:
                    pygame.draw.rect(self.screen, (255, 255, 255), 
                                   (center_x - line_width // 2, dash_start, 
                                    line_width, dash_end - dash_start))
            y += total_cycle
        
        # Edge lines with enhanced offset
        edge_color = (255, 255, 0)
        left_edge_x = road_x
        right_edge_x = road_x + road_width - line_width
        
        if left_edge_x >= 0 and left_edge_x <= self.screen_width:
            pygame.draw.rect(self.screen, edge_color, (left_edge_x, 0, line_width, self.screen_height))
        if right_edge_x >= 0 and right_edge_x <= self.screen_width:
            pygame.draw.rect(self.screen, edge_color, (right_edge_x, 0, line_width, self.screen_height))
    
    def update_obstacles(self, current_time, delta_time):
        """Update obstacles with enhanced horizontal world offset"""
        # Create new obstacles
        if self.settings["obstacle_frequency"] > 0:
            if current_time >= self.next_obstacle_time:
                # Create obstacle at world position with current offset
                obstacle_world_x = random.randint(100, self.screen_width - 100) + self.road_offset_x
                obstacle = Obstacle(obstacle_world_x, -50, speed=self.settings["obstacle_speed"])
                obstacle.world_x = obstacle_world_x  # Store world position
                self.obstacles.append(obstacle)
                
                self.next_obstacle_time = current_time + (1.0 / self.settings["obstacle_frequency"])
        
        # Update existing obstacles
        for obstacle in self.obstacles[:]:
            obstacle.update(delta_time)
            
            # וודא שלכל מכשול יש תכונת world_x
            if not hasattr(obstacle, 'world_x'):
                obstacle.world_x = obstacle.x
            
            # Remove off-screen obstacles (considering enhanced horizontal offset)
            screen_x = obstacle.world_x - self.road_offset_x
            if obstacle.y > self.screen_height + 50 or screen_x < -150 or screen_x > self.screen_width + 150:
                self.obstacles.remove(obstacle)
                
                # Score for avoiding obstacle
                if not obstacle.hit:
                    points = SCORE_PER_OBSTACLE * self.settings["score_multiplier"]
                    self.score += points

    def process_camera_input(self):
        """Process camera input for gesture detection"""
        if self.camera is None or self.hand_detector is None:
            return None
        
        try:
            success, self.frame = read_frame(self.camera)
            
            if not success or self.frame is None:
                return None
            
            self.frame, results = self.hand_detector.find_hands(self.frame)
            landmarks = self.hand_detector.find_positions(self.frame, results)
            
            if landmarks:
                gestures = self.gesture_recognizer.recognize_gestures(landmarks, self.frame.shape[0])
                return gestures
        except Exception as e:
            logger.error(f"Error processing camera input: {e}")
        
        return None

    def draw_camera_feed(self):
        """Draw camera feed display"""
        if self.frame is not None and self.show_camera:
            try:
                # Resize frame for display
                display_width = self.frame.shape[1] // 4
                display_height = self.frame.shape[0] // 4
                
                display_frame = cv2.resize(self.frame, (display_width, display_height))
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
                # Create pygame surface
                feed_surface = pygame.surfarray.make_surface(display_frame.swapaxes(0, 1))
                
                # Draw with border
                feed_rect = pygame.Rect(self.screen_width, 0, display_width, display_height)
                pygame.draw.rect(self.screen, WHITE, feed_rect, width=2)
                self.screen.blit(feed_surface, feed_rect)
            except Exception as e:
                logger.error(f"Error drawing camera feed: {e}")
        elif self.show_camera:
            # Camera not available message
            camera_rect = pygame.Rect(self.screen_width, 0, 320, 240)
            pygame.draw.rect(self.screen, (30, 30, 30), camera_rect)
            
            messages = [
                "Camera Not Available",
                "Using Keyboard Controls:",
                "Arrow Keys / WASD: Movement",
                "Space: Boost",
                "FIXED VERSION ✅"
            ]
            
            for i, msg in enumerate(messages):
                font_size = 16 if i > 0 else 20
                font = pygame.font.SysFont(None, font_size)
                if i == 0:
                    color = (255, 50, 50)  # Red for error
                elif i == 4:
                    color = (0, 255, 0)   # Green for fixed
                else:
                    color = (200, 200, 200)  # Gray for normal
                    
                text = font.render(msg, True, color)
                self.screen.blit(text, (self.screen_width + 10, 50 + i * 25))
                
    def cleanup(self):
        """Clean up resources before exiting"""
        # Camera cleanup
        if self.camera is not None:
            try:
                self.camera.release()
                logger.info("Camera resources released")
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")
        
        # Hand detector cleanup
        if self.hand_detector is not None:
            try:
                self.hand_detector.release()
                logger.info("Hand detector resources released")
            except Exception as e:
                logger.error(f"Error releasing hand detector: {e}")
        
        # Sound cleanup
        try:
            self.sound_manager.cleanup()
            logger.info("Sound resources released")
        except Exception as e:
            logger.error(f"Error cleaning up sound: {e}")
        
        # OpenCV cleanup
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            logger.error(f"Error destroying OpenCV windows: {e}")
            
        logger.info("All resources cleaned up")

    def draw(self):
        """Draw the complete game screen with enhanced steering background movement"""
        # Clear screen
        self.screen.fill((50, 50, 50))
        
        # Draw road (MovingRoad if available, otherwise built-in)
        if self.moving_road:
            try:
                self.moving_road.draw(self.screen)
            except Exception as e:
                logger.error(f"Error drawing moving road: {e}")
                self.draw_built_in_road()
        else:
            self.draw_built_in_road()
        
        # Draw game objects with enhanced world offset
        world_offset_x = int(self.road_offset_x)
        for obstacle in self.obstacles:
            # Calculate screen position for obstacle with enhanced effect
            if hasattr(obstacle, 'world_x'):
                screen_x = obstacle.world_x - world_offset_x
            else:
                # Fallback if world_x is missing
                obstacle.world_x = obstacle.x  # Ensure it has world_x property
                screen_x = obstacle.world_x - world_offset_x
                
            screen_y = obstacle.y
            
            # Only draw if on screen with wider margins
            if -100 <= screen_x <= self.screen_width + 100:
                # Temporarily set obstacle position for drawing
                original_x = obstacle.x
                obstacle.x = screen_x
                obstacle.draw(self.screen)
                obstacle.x = original_x  # Restore world position
        
        # Car always stays in center
        self.car.draw(self.screen)
        
        # Draw UI elements
        self.draw_ui()
        self.draw_camera_feed()
        
        # Draw overlays
        if self.paused:
            self.draw_pause_menu()
        
        if self.game_over or self.game_completed:
            self.draw_game_over()
            
        if self.debug_mode:
            self.draw_debug_info()
        
        if self.show_help:
            self.draw_help_text()
        
        # Update display
        pygame.display.flip()

    def draw_ui(self):
        """Draw UI elements like buttons and status"""
        # Draw buttons
        self.pause_button.draw(self.screen)
        self.mute_button.draw(self.screen)
        
        # Draw score
        score_text = self.font.render(f"Score: {int(self.score)}", True, WHITE)
        self.screen.blit(score_text, (20, 20))
        
        # Draw speed indicator
        speed_text = self.font.render(f"Speed: {self.car.speed:.1f}", True, WHITE)
        self.screen.blit(speed_text, (20, 50))
        
        # Draw time remaining if applicable
        if self.time_left > 0:
            minutes = int(self.time_left) // 60
            seconds = int(self.time_left) % 60
            time_text = self.font.render(f"Time: {minutes}:{seconds:02d}", True, WHITE)
            self.screen.blit(time_text, (self.screen_width // 2 - 50, 20))

    def draw_pause_menu(self):
        """Draw pause menu overlay"""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.set_alpha(128)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        # Pause text
        pause_text = self.title_font.render("PAUSED", True, WHITE)
        text_rect = pause_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        self.screen.blit(pause_text, text_rect)
        
        # Instructions
        inst_text = self.font.render("Press P to resume", True, WHITE)
        inst_rect = inst_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 50))
        self.screen.blit(inst_text, inst_rect)

    def draw_game_over(self):
        """Draw game over screen"""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.set_alpha(180)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        # Game over text
        if self.game_completed:
            title = "GAME COMPLETED!"
            color = SUCCESS
        else:
            title = "GAME OVER"
            color = ERROR
            
        title_text = self.title_font.render(title, True, color)
        title_rect = title_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 50))
        self.screen.blit(title_text, title_rect)
        
        # Final score
        score_text = self.font.render(f"Final Score: {int(self.score)}", True, WHITE)
        score_rect = score_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        self.screen.blit(score_text, score_rect)
        
        # Instructions
        inst_text = self.font.render("Press ESC to exit", True, WHITE)
        inst_rect = inst_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 50))
        self.screen.blit(inst_text, inst_rect)

    def draw_debug_info(self):
        """Draw debug information"""
        debug_info = [
            f"Car Pos: ({self.car.x:.1f}, {self.car.y:.1f})",
            f"Car Rotation: {self.car.rotation:.1f}°",
            f"Road Offset X: {self.road_offset_x:.1f}",
            f"Obstacles: {len(self.obstacles)}",
            f"FPS: {self.clock.get_fps():.1f}"
        ]
        
        debug_y = self.screen_height - 150
        for i, info in enumerate(debug_info):
            debug_text = self.font.render(info, True, WHITE)
            self.screen.blit(debug_text, (20, debug_y + i * 25))

    def draw_help_text(self):
        """Draw help text overlay"""
        # Help panel background
        help_rect = pygame.Rect(self.screen_width - 250, 120, 240, 200)
        pygame.draw.rect(self.screen, (30, 30, 30, 200), help_rect)
        pygame.draw.rect(self.screen, WHITE, help_rect, 2)
        
        help_lines = [
            "Controls:",
            "Arrow Keys - Steer/Speed",
            "WASD - Alternative",
            "Space - Boost",
            "H - Toggle Help",
            "D - Toggle Debug",
            "C - Toggle Camera",
            "M - Mute/Unmute",
            "ESC - Exit"
        ]
        
        for i, line in enumerate(help_lines):
            color = ACCENT if i == 0 else WHITE
            help_text = pygame.font.Font(None, 18).render(line, True, color)
            self.screen.blit(help_text, (help_rect.x + 10, help_rect.y + 10 + i * 20))

    def _confirm_exit(self):
        """Show confirmation dialog for exit"""
        # For now, just return True. In future versions, could show actual dialog
        return True

    def run(self):
        """Main game loop - SYNCHRONIZED VERSION"""
        self.running = True
        last_time = time.time()
        
        logger.info("🏁 Starting CLEAN game loop with FIXED road direction")
        
        # Ensure display is ready
        if not pygame.display.get_surface():
            print("⚠️ Recreating display surface")
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width + 320, self.screen_height))
        
        # Initialize timing
        self.start_time = time.time()
        self.last_position = (self.car.x, self.car.y)
        
        print(f"🎮 FIXED VERSION - Road animation: {'MovingRoad' if self.moving_road else 'Built-in'}")
        
        try:
            while self.running:
                # Calculate frame time
                current_time = time.time()
                dt = min(current_time - last_time, 0.1)  # Cap at 100ms
                last_time = current_time
                
                # Process events
                self.handle_events()
                
                # Update game logic
                self.update(dt)
                
                # Render frame
                self.draw()
                
                # Maintain frame rate
                self.clock.tick(self.target_fps)
        
        except Exception as e:
            logger.error(f"Error in game loop: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Always clean up
            self.cleanup()
            
            if pygame.get_init():
                pygame.quit()

# Add this at end of file to expose the run_game function
def run_game(mode="normal", hand_detector=None, show_tutorial=True, config=None):
    """
    Run the game - SYNCHRONIZED VERSION
    
    Args:
        mode: Game mode ("easy", "normal", "hard")
        hand_detector: Hand gesture detector instance (optional)
        show_tutorial: Whether to show tutorial (ignored for now)
        config: Optional configuration (ignored for now)
        
    Returns:
        Final score achieved
    """
    print(f"🎮 FIXED VERSION - Starting game in {mode} mode")
    
    # Create game instance
    game = Game(mode, hand_detector=hand_detector)
    
    try:
        # Run the game
        game.running = True
        game.run()
        
        # Return final score
        return game.score
    except Exception as e:
        print(f"❌ Error running game: {e}")
        import traceback
        traceback.print_exc()
        return 0
    finally:
        # Ensure cleanup is called if it hasn't been already
        try:
            game.cleanup()
        except:
            pass

# Add the export of run_game to module exports
__all__ = ['Game', 'run_game']