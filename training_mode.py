#!/usr/bin/env python
"""
Training Mode for Car Control with Hand Gestures

This module implements a training mode for the car control game, where:
- The car starts in the center of the road
- The player can practice maneuvering using hand gestures
- Static obstacles are placed for practice
- Visual feedback is provided to help learn the control system

This training mode is designed to help users get comfortable with the
hand gesture controls before playing the main game modes.
"""

import pygame
import sys
import os
import random
import math
import time
import numpy as np
import cv2
from pygame.locals import *

# Import our hand detection module
from hand_detector.improved_hand_gesture_detector import EnhancedHandGestureDetector

# Import game configuration
from config import GAME_MODES

# Constants for the game
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (100, 100, 100)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
ROAD_COLOR = (80, 80, 80)
LANE_COLOR = (240, 240, 240)

# Training mode specific constants
OBSTACLE_TYPES = [
    {"name": "cone", "width": 20, "height": 30, "color": (255, 140, 0)},
    {"name": "barrier", "width": 80, "height": 15, "color": (255, 50, 50)},
    {"name": "rock", "width": 40, "height": 40, "color": (120, 120, 120)}
]

def get_available_cameras():
    """Get a list of available camera devices on the system."""
    available_cameras = []
    camera_index = 0
    
    # Try to open cameras until we find one that doesn't work
    while True:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            break
            
        # Test if the camera is working by trying to read a frame
        ret, _ = cap.read()
        if ret:
            # Camera works, add to list
            name = f"Camera {camera_index}"
            try:
                # Try to get camera name (works on some systems)
                backend = cap.getBackendName()
                if backend:
                    name = f"Camera {camera_index} ({backend})"
            except:
                pass
            
            available_cameras.append({"index": camera_index, "name": name})
        
        # Release the camera and try next index
        cap.release()
        camera_index += 1
        
        # Safety check to avoid infinite loop
        if camera_index > 10:  # Assume max 10 cameras
            break
    
    return available_cameras

class Car:
    """Represents the player's car in the game."""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 40
        self.height = 80
        self.direction = 0  # -1 to 1 (left to right)
        self.speed = 0  # 0 to 1
        self.max_speed = 300  # Maximum speed in pixels per second
        self.boost_multiplier = 1.5  # Speed multiplier when boost is active
        self.boost_active = False
        self.braking = False
        self.color = (50, 50, 200)  # Blue car
        
        # Control smoothing
        self.target_direction = 0
        self.target_speed = 0
        self.direction_smoothing = 0.2
        self.speed_smoothing = 0.1
        
        # Collision detection
        self.collision_radius = 40
        self.collision_points = []
        self.update_collision_points()
        
        # Visual indicators
        self.show_indicators = True
        
    def update(self, controls, dt):
        """Update car position and state based on controls and time delta."""
        # Extract controls (steering, throttle, braking, boost)
        self.target_direction = float(controls.get('steering', 0)) 
        self.target_speed = float(controls.get('throttle', 0))
        self.braking = bool(controls.get('braking', False))
        self.boost_active = bool(controls.get('boost', False))
        
        # Apply smoothing to controls for more natural movement
        self.direction = self.lerp(self.direction, self.target_direction, self.direction_smoothing)
        
        # Handle braking/boosting
        if self.braking:
            # Apply brakes (rapid deceleration)
            self.speed = self.lerp(self.speed, 0, 0.3)  # Faster reduction for braking
        elif self.boost_active:
            # Apply boost (rapid acceleration to max)
            self.speed = self.lerp(self.speed, self.target_speed, 0.2)  # Faster boost application
        else:
            # Normal acceleration/deceleration
            self.speed = self.lerp(self.speed, self.target_speed, self.speed_smoothing)
        
        # Move the car based on speed and direction
        speed_pixels_per_second = self.max_speed * self.speed
        if self.boost_active:
            speed_pixels_per_second *= self.boost_multiplier
            
        # Calculate movement
        move_amount = speed_pixels_per_second * dt
        turn_factor = self.direction * move_amount * 0.05
        
        # Update car position - in training mode movement is only horizontal
        self.x += turn_factor * 10  # Amplified horizontal movement for better control practice
        
        # Keep car within screen bounds
        self.x = max(self.width//2, min(SCREEN_WIDTH - self.width//2, self.x))
        
        # Update collision detection points
        self.update_collision_points()
        
    def lerp(self, current, target, factor):
        """Linear interpolation for smoother movement."""
        return current + (target - current) * factor
        
    def draw(self, screen):
        """Draw the car on the screen with visual indicators."""
        # Draw car body
        car_rect = pygame.Rect(
            self.x - self.width//2,
            self.y - self.height//2,
            self.width,
            self.height
        )
        pygame.draw.rect(screen, self.color, car_rect, 0, 10)  # Rounded corners
        
        # Draw car details (windshield, lights)
        # Windshield
        windshield_rect = pygame.Rect(
            self.x - self.width//3,
            self.y - self.height//2 + 10,
            self.width//1.5,
            self.height//3
        )
        pygame.draw.rect(screen, (150, 220, 255), windshield_rect, 0, 5)
        
        # Front lights
        light_size = 8
        # Left light
        pygame.draw.circle(screen, YELLOW, 
                         (int(self.x - self.width//4), int(self.y - self.height//2 + 5)), 
                         light_size)
        # Right light
        pygame.draw.circle(screen, YELLOW, 
                         (int(self.x + self.width//4), int(self.y - self.height//2 + 5)), 
                         light_size)
        
        # Draw special effects based on car state
        if self.boost_active:
            # Draw flame effect for boost
            flame_points = [
                (self.x - 10, self.y + self.height//2 + 10),
                (self.x, self.y + self.height//2 + 25),
                (self.x + 10, self.y + self.height//2 + 10)
            ]
            pygame.draw.polygon(screen, (255, 165, 0), flame_points)
            
        if self.braking:
            # Draw brake lights
            brake_light_y = self.y + self.height//2 - 5
            # Left brake light
            pygame.draw.circle(screen, RED, 
                            (int(self.x - self.width//4), int(brake_light_y)), 
                            light_size)
            # Right brake light
            pygame.draw.circle(screen, RED, 
                            (int(self.x + self.width//4), int(brake_light_y)), 
                            light_size)
        
        # Draw visual indicators if enabled
        if self.show_indicators:
            # Direction indicator (steering)
            indicator_length = 50
            steer_angle = -self.direction * 30  # Convert steering (-1 to 1) to angle
            end_x = self.x + indicator_length * math.sin(math.radians(steer_angle))
            end_y = self.y - indicator_length * math.cos(math.radians(steer_angle))
            pygame.draw.line(screen, GREEN, (self.x, self.y), (end_x, end_y), 2)
            
            # Speed indicator
            speed_height = 40 * self.speed  # Scale with speed
            speed_rect = pygame.Rect(
                self.x - self.width//2 - 20,
                self.y + self.height//2 - speed_height,
                10,
                speed_height
            )
            speed_color = GREEN if not self.boost_active else (255, 165, 0)
            pygame.draw.rect(screen, speed_color, speed_rect)
            
            # Draw text labels for indicators
            font = pygame.font.Font(None, 20)
            # Steering label
            steer_text = font.render(f"Steer: {self.direction:.2f}", True, WHITE)
            screen.blit(steer_text, (self.x - self.width//2 - 80, self.y - 30))
            # Speed label
            speed_text = font.render(f"Speed: {self.speed:.2f}", True, WHITE)
            screen.blit(speed_text, (self.x - self.width//2 - 80, self.y))
            
    def update_collision_points(self):
        """Update collision detection points around the car."""
        # Create points around the car for more accurate collision detection
        self.collision_points = [
            # Front center
            (self.x, self.y - self.height//2),
            # Front left corner
            (self.x - self.width//2, self.y - self.height//2),
            # Front right corner
            (self.x + self.width//2, self.y - self.height//2),
            # Rear center
            (self.x, self.y + self.height//2),
            # Center point
            (self.x, self.y)
        ]
        
    def check_collision(self, obstacle):
        """Check if the car collides with the given obstacle."""
        # Create obstacle rectangle
        obstacle_rect = pygame.Rect(
            obstacle['x'] - obstacle['width']//2,
            obstacle['y'] - obstacle['height']//2,
            obstacle['width'],
            obstacle['height']
        )
        
        # Check if any of the collision points are within the obstacle
        for point in self.collision_points:
            if obstacle_rect.collidepoint(point):
                return True
                
        # Also check car rectangle for complete coverage
        car_rect = pygame.Rect(
            self.x - self.width//2,
            self.y - self.height//2,
            self.width,
            self.height
        )
        
        return car_rect.colliderect(obstacle_rect)


class TrainingMode:
    """Training mode for practicing car control with hand gestures."""
    
    def __init__(self):
        """Initialize the training mode."""
        pygame.init()
        pygame.display.set_caption("Car Control Training Mode")
        
        # Set up the screen
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Set up fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Create the player's car in the center of the road
        self.car = Car(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 100)
        
        # Create fixed obstacles for training
        self.obstacles = self.create_training_obstacles()
        
        # Set up the road
        self.road_width = 600
        self.lane_width = self.road_width // 3
        self.road_x = (SCREEN_WIDTH - self.road_width) // 2
        
        # Scoring and game state
        self.score = 0
        self.start_time = 0
        self.game_active = False
        self.game_over = False
        self.show_tutorial = True
        self.tutorial_step = 0
        self.tutorial_steps = [
            "Welcome to Training Mode! Let's learn to control the car with hand gestures.",
            "Move your hand left and right to steer the car.",
            "Move your hand up and down to control speed.",
            "Make a fist to brake. Keep your fist raised for boost!",
            "Show an open palm (stop sign) to emergency stop.",
            "Let's start practicing! Navigate around the obstacles.",
            "Ready to begin? Press SPACE to start."
        ]
        
        # Set up camera selection
        self.available_cameras = get_available_cameras()
        self.current_camera_index = 0 if self.available_cameras else -1
        self.show_camera_selection = len(self.available_cameras) > 1
        self.camera_enabled = self.current_camera_index >= 0
        self.cap = None
        
        # Initialize camera if available
        if self.camera_enabled:
            self.select_camera(self.current_camera_index)
        
        # Camera display settings
        self.camera_display_size = (320, 240)
        self.show_camera = True
        self.camera_frame = None
        self.data_panel = None
        
    def select_camera(self, camera_index):
        """Select and initialize a camera by index."""
        # Release current camera if any
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        # Try to initialize the selected camera
        try:
            if 0 <= camera_index < len(self.available_cameras):
                index = self.available_cameras[camera_index]["index"]
                self.cap = cv2.VideoCapture(index)
                
                # Check if camera opened successfully
                if not self.cap.isOpened():
                    print(f"Error: Could not open camera {index}.")
                    self.camera_enabled = False
                    return False
                else:
                    # Set camera resolution
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.current_camera_index = camera_index
                    self.camera_enabled = True
                    
                    # Initialize the hand detector with the new camera
                    try:
                        self.detector = EnhancedHandGestureDetector()
                    except Exception as e:
                        print(f"Error initializing hand detector: {e}")
                        self.camera_enabled = False
                        return False
                        
                    return True
            else:
                print(f"Invalid camera index: {camera_index}")
                self.camera_enabled = False
                return False
                
        except Exception as e:
            print(f"Error selecting camera: {e}")
            self.camera_enabled = False
            return False
            
    def create_training_obstacles(self):
        """Create a fixed pattern of obstacles for training purposes."""
        obstacles = []
        
        # Create a slalom course with alternating obstacles
        num_obstacles = 8
        obstacle_spacing = SCREEN_HEIGHT / (num_obstacles + 1)
        
        for i in range(num_obstacles):
            # Alternate obstacles between left and right side of the road
            x_position = SCREEN_WIDTH // 4 if i % 2 == 0 else SCREEN_WIDTH * 3 // 4
            y_position = obstacle_spacing * (i + 1)
            
            # Select random obstacle type
            obstacle_type = random.choice(OBSTACLE_TYPES)
            
            obstacles.append({
                'x': x_position,
                'y': y_position,
                'width': obstacle_type['width'],
                'height': obstacle_type['height'],
                'color': obstacle_type['color'],
                'type': obstacle_type['name']
            })
        
        # Add some random obstacles in the middle lane
        for _ in range(3):
            x_position = SCREEN_WIDTH // 2 + random.randint(-50, 50)
            y_position = random.randint(100, SCREEN_HEIGHT - 200)
            
            # Ensure obstacles aren't too close to each other
            too_close = False
            for obs in obstacles:
                dist = math.sqrt((x_position - obs['x'])**2 + (y_position - obs['y'])**2)
                if dist < 100:
                    too_close = True
                    break
            
            if not too_close:
                obstacle_type = random.choice(OBSTACLE_TYPES)
                obstacles.append({
                    'x': x_position,
                    'y': y_position,
                    'width': obstacle_type['width'],
                    'height': obstacle_type['height'],
                    'color': obstacle_type['color'],
                    'type': obstacle_type['name']
                })
        
        return obstacles
        
    def process_hand_gestures(self):
        """Process hand gestures from the camera for car control."""
        if not self.camera_enabled or self.cap is None:
            return {
                'steering': 0.0,
                'throttle': 0.0,
                'braking': False,
                'boost': False
            }
        
        try:
            # Read frame from the camera
            ret, frame = self.cap.read()
            
            if not ret:
                print("Error reading frame from camera.")
                return {
                    'steering': 0.0,
                    'throttle': 0.0,
                    'braking': False,
                    'boost': False
                }
            
            # Flip the frame for mirror effect (more intuitive)
            frame = cv2.flip(frame, 1)
            
            # Process the frame for hand gestures
            controls, processed_frame, data_panel = self.detector.detect_gestures(frame)
            
            # Store the processed frame and data panel for display
            self.camera_frame = cv2.resize(processed_frame, self.camera_display_size)
            self.data_panel = cv2.resize(data_panel, (self.camera_display_size[0], self.camera_display_size[1]))
            
            return controls
            
        except Exception as e:
            print(f"Error processing hand gestures: {e}")
            return {
                'steering': 0.0,
                'throttle': 0.0,
                'braking': False,
                'boost': False
            }
    
    def handle_events(self):
        """Handle game events including user input."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_SPACE:
                    if self.show_camera_selection:
                        # Confirm camera selection
                        self.show_camera_selection = False
                    elif self.show_tutorial:
                        # Advance tutorial or start game
                        self.tutorial_step += 1
                        if self.tutorial_step >= len(self.tutorial_steps):
                            self.show_tutorial = False
                            self.start_game()
                    elif self.game_over:
                        # Restart game
                        self.__init__()
                    elif not self.game_active:
                        # Start game
                        self.start_game()
                        
                # Toggle camera display with 'c' key
                elif event.key == pygame.K_c:
                    self.show_camera = not self.show_camera
                    
                # Restart with 'r' key
                elif event.key == pygame.K_r:
                    self.__init__()
                    
                # Camera switching with left/right arrow keys
                elif event.key in (pygame.K_LEFT, pygame.K_RIGHT) and len(self.available_cameras) > 1:
                    if self.show_camera_selection or pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        # Only allow camera switching in selection screen or with Shift key
                        direction = -1 if event.key == pygame.K_LEFT else 1
                        new_index = (self.current_camera_index + direction) % len(self.available_cameras)
                        self.select_camera(new_index)
                    
        return True
        
    def start_game(self):
        """Start the training mode game."""
        self.game_active = True
        self.start_time = time.time()
        self.score = 0
        
    def update(self):
        """Update game state for one frame."""
        # Get the time elapsed since last frame
        dt = self.clock.get_time() / 1000.0  # Convert to seconds
        
        if self.game_active and not self.game_over:
            # Process hand gestures for car control
            controls = self.process_hand_gestures()
            
            # Update car position and state
            self.car.update(controls, dt)
            
            # Check for collisions with obstacles
            for obstacle in self.obstacles:
                if self.car.check_collision(obstacle):
                    # Handle collision - in training mode, just highlight the obstacle
                    obstacle['hit'] = True
                    # Add a small penalty to score
                    self.score -= 10
            
            # Update score based on time (1 point per second)
            elapsed_time = time.time() - self.start_time
            self.score = int(elapsed_time * 10)  # 10 points per second
    
    def draw(self):
        """Draw the game on the screen."""
        # Fill background
        self.screen.fill(BLACK)
        
        # If we're showing the camera selection screen
        if self.show_camera_selection:
            self.draw_camera_selection()
            pygame.display.flip()
            return
        
        # Draw road
        self.draw_road()
        
        # Draw obstacles
        self.draw_obstacles()
        
        # Draw the player's car
        self.car.draw(self.screen)
        
        # Draw HUD
        self.draw_hud()
        
        # Draw camera feed if enabled
        if self.show_camera and self.camera_frame is not None:
            self.draw_camera_feed()
        
        # Draw tutorial if active
        if self.show_tutorial:
            self.draw_tutorial()
        
        # Draw game over screen if game is over
        if self.game_over:
            self.draw_game_over()
            
        # Update the display
        pygame.display.flip()
    
    def draw_road(self):
        """Draw the road and lanes."""
        # Draw the main road
        road_rect = pygame.Rect(self.road_x, 0, self.road_width, SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, ROAD_COLOR, road_rect)
        
        # Draw lane markers
        for i in range(1, 3):
            lane_x = self.road_x + i * self.lane_width
            
            # Draw dashed lines for lanes
            dash_length = 30
            gap_length = 20
            total_length = dash_length + gap_length
            
            for y in range(0, SCREEN_HEIGHT, total_length):
                pygame.draw.line(
                    self.screen,
                    LANE_COLOR,
                    (lane_x, y),
                    (lane_x, y + dash_length),
                    3
                )
    
    def draw_obstacles(self):
        """Draw the obstacles on the road."""
        for obstacle in self.obstacles:
            # Determine color - highlight if hit
            color = RED if obstacle.get('hit', False) else obstacle['color']
            
            # Draw the obstacle
            obstacle_rect = pygame.Rect(
                obstacle['x'] - obstacle['width']//2,
                obstacle['y'] - obstacle['height']//2,
                obstacle['width'],
                obstacle['height']
            )
            
            # Draw differently based on obstacle type
            if obstacle['type'] == 'cone':
                # Draw a triangle for cones
                points = [
                    (obstacle['x'], obstacle['y'] - obstacle['height']//2),
                    (obstacle['x'] - obstacle['width']//2, obstacle['y'] + obstacle['height']//2),
                    (obstacle['x'] + obstacle['width']//2, obstacle['y'] + obstacle['height']//2)
                ]
                pygame.draw.polygon(self.screen, color, points)
                
                # Add a white stripe
                pygame.draw.line(
                    self.screen,
                    WHITE,
                    (obstacle['x'] - obstacle['width']//4, obstacle['y']),
                    (obstacle['x'] + obstacle['width']//4, obstacle['y']),
                    2
                )
                
            elif obstacle['type'] == 'barrier':
                # Draw a rectangle with stripes for barriers
                pygame.draw.rect(self.screen, color, obstacle_rect)
                
                # Add alternating white/red stripes
                stripe_width = obstacle['width'] // 4
                for i in range(4):
                    stripe_color = WHITE if i % 2 == 0 else RED
                    stripe_rect = pygame.Rect(
                        obstacle['x'] - obstacle['width']//2 + i * stripe_width,
                        obstacle['y'] - obstacle['height']//2,
                        stripe_width,
                        obstacle['height']
                    )
                    pygame.draw.rect(self.screen, stripe_color, stripe_rect)
                
            else:  # default/rock
                # Draw a circle for rocks/generic obstacles
                pygame.draw.circle(
                    self.screen,
                    color,
                    (obstacle['x'], obstacle['y']),
                    obstacle['width'] // 2
                )
                
                # Add some texture
                pygame.draw.circle(
                    self.screen,
                    (color[0] - 30, color[1] - 30, color[2] - 30),
                    (obstacle['x'] + 5, obstacle['y'] - 5),
                    obstacle['width'] // 4
                )
    
    def draw_hud(self):
        """Draw heads-up display with game information."""
        # Draw score and time
        if self.game_active:
            elapsed_time = time.time() - self.start_time
            score_text = self.font_medium.render(f"Score: {self.score}", True, WHITE)
            time_text = self.font_medium.render(f"Time: {int(elapsed_time)}s", True, WHITE)
            
            self.screen.blit(score_text, (20, 20))
            self.screen.blit(time_text, (20, 60))
            
            # Draw current camera info if multiple cameras available
            if len(self.available_cameras) > 1:
                camera_name = self.available_cameras[self.current_camera_index]["name"]
                camera_text = self.font_small.render(f"Camera: {camera_name} (Shift+←/→ to switch)", True, WHITE)
                self.screen.blit(camera_text, (20, 100))
            
            # Draw controls guide
            control_text = self.font_small.render("Hand Left/Right: Steer | Hand Up/Down: Speed | Fist: Brake | Fist Up: Boost | Open Palm: Stop", True, WHITE)
            self.screen.blit(control_text, (SCREEN_WIDTH // 2 - control_text.get_width() // 2, SCREEN_HEIGHT - 30))
    
    def draw_camera_feed(self):
        """Draw the camera feed and data panel."""
        if self.camera_frame is not None and self.data_panel is not None:
            # Get the actual dimensions of the camera frame
            h, w = self.camera_frame.shape[:2]
            
            # Convert OpenCV BGR format to RGB for Pygame
            camera_rgb = cv2.cvtColor(self.camera_frame, cv2.COLOR_BGR2RGB)
            camera_surface = pygame.Surface((w, h))
            pygame_camera = pygame.image.frombuffer(camera_rgb.tobytes(), (w, h), "RGB")
            camera_surface.blit(pygame_camera, (0, 0))
            
            # Same for data panel
            h_panel, w_panel = self.data_panel.shape[:2]
            panel_rgb = cv2.cvtColor(self.data_panel, cv2.COLOR_BGR2RGB)
            panel_surface = pygame.Surface((w_panel, h_panel))
            pygame_panel = pygame.image.frombuffer(panel_rgb.tobytes(), (w_panel, h_panel), "RGB")
            panel_surface.blit(pygame_panel, (0, 0))
            
            # Draw camera feed in top-right corner
            self.screen.blit(camera_surface, (SCREEN_WIDTH - w, 0))
            
            # Draw data panel below camera feed
            self.screen.blit(panel_surface, (SCREEN_WIDTH - w_panel, h + 10))
            
            # Add label
            camera_name = self.available_cameras[self.current_camera_index]["name"] if self.available_cameras else "Unknown"
            label_text = self.font_small.render(f"Hand Detection - {camera_name}", True, WHITE)
            self.screen.blit(label_text, (SCREEN_WIDTH - w + 10, 10))
            
    def draw_camera_selection(self):
        """Draw the camera selection screen."""
        # Create a background
        self.screen.fill((30, 30, 50))
        
        # Draw title
        title_text = self.font_large.render("Select a Camera", True, WHITE)
        self.screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() // 2, 100))
        
        if not self.available_cameras:
            # No cameras available
            no_cam_text = self.font_medium.render("No cameras detected", True, RED)
            self.screen.blit(no_cam_text, (SCREEN_WIDTH // 2 - no_cam_text.get_width() // 2, 250))
            
            instructions = self.font_small.render("Press SPACE to continue without camera", True, WHITE)
            self.screen.blit(instructions, (SCREEN_WIDTH // 2 - instructions.get_width() // 2, 300))
        else:
            # Show available cameras
            y_pos = 200
            for i, camera in enumerate(self.available_cameras):
                # Highlight current selection
                color = GREEN if i == self.current_camera_index else WHITE
                
                camera_text = self.font_medium.render(camera["name"], True, color)
                text_rect = camera_text.get_rect(center=(SCREEN_WIDTH // 2, y_pos))
                
                # Draw selection indicator
                if i == self.current_camera_index:
                    pygame.draw.rect(
                        self.screen,
                        GREEN,
                        (text_rect.x - 20, text_rect.y - 5, text_rect.width + 40, text_rect.height + 10),
                        2,
                        5
                    )
                
                self.screen.blit(camera_text, text_rect)
                y_pos += 50
            
            # Preview selected camera
            if self.camera_enabled and self.cap is not None:
                try:
                    # Read a frame from the camera
                    ret, frame = self.cap.read()
                    if ret:
                        # Flip and resize frame for display
                        frame = cv2.flip(frame, 1)
                        frame = cv2.resize(frame, (320, 240))
                        
                        # Convert to pygame surface
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        preview_surface = pygame.Surface((320, 240))
                        preview_frame = pygame.image.frombuffer(frame_rgb.tobytes(), (320, 240), "RGB")
                        preview_surface.blit(preview_frame, (0, 0))
                        
                        # Draw preview
                        preview_x = SCREEN_WIDTH // 2 - 160
                        preview_y = 350
                        self.screen.blit(preview_surface, (preview_x, preview_y))
                        
                        # Draw border around preview
                        pygame.draw.rect(
                            self.screen,
                            GREEN,
                            (preview_x - 2, preview_y - 2, 324, 244),
                            2
                        )
                        
                        # Draw preview label
                        preview_label = self.font_small.render("Camera Preview", True, WHITE)
                        self.screen.blit(preview_label, (SCREEN_WIDTH // 2 - preview_label.get_width() // 2, preview_y - 30))
                except:
                    # Handle any errors with camera preview
                    error_text = self.font_small.render("Error displaying camera preview", True, RED)
                    self.screen.blit(error_text, (SCREEN_WIDTH // 2 - error_text.get_width() // 2, 400))
            
            # Instructions
            instructions = self.font_small.render("Use LEFT/RIGHT arrow keys to select, SPACE to confirm", True, WHITE)
            self.screen.blit(instructions, (SCREEN_WIDTH // 2 - instructions.get_width() // 2, SCREEN_HEIGHT - 50))
    
    def draw_tutorial(self):
        """Draw the tutorial overlay."""
        # Create a semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))  # Black with transparency
        self.screen.blit(overlay, (0, 0))
        
        # Draw title
        title_text = self.font_large.render("Training Mode Tutorial", True, WHITE)
        self.screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() // 2, 100))
        
        # Draw current tutorial step
        step_text = self.font_medium.render(self.tutorial_steps[self.tutorial_step], True, WHITE)
        # Word wrap for long text
        words = self.tutorial_steps[self.tutorial_step].split()
        lines = []
        line = []
        line_width = 0
        for word in words:
            word_surface = self.font_medium.render(word + " ", True, WHITE)
            word_width = word_surface.get_width()
            if line_width + word_width <= SCREEN_WIDTH - 100:
                line.append(word)
                line_width += word_width
            else:
                lines.append(" ".join(line))
                line = [word]
                line_width = word_width
        if line:
            lines.append(" ".join(line))
        
        # Draw each line of text
        y_offset = 200
        for line in lines:
            line_surface = self.font_medium.render(line, True, WHITE)
            self.screen.blit(line_surface, (SCREEN_WIDTH // 2 - line_surface.get_width() // 2, y_offset))
            y_offset += 40
        
        # Draw progress indicator
        for i in range(len(self.tutorial_steps)):
            color = GREEN if i == self.tutorial_step else LIGHT_GRAY
            pygame.draw.circle(self.screen, color, (SCREEN_WIDTH // 2 - (len(self.tutorial_steps) * 15) // 2 + i * 30, 350), 10)
        
        # Draw navigation hint
        nav_text = self.font_small.render("Press SPACE to continue", True, WHITE)
        self.screen.blit(nav_text, (SCREEN_WIDTH // 2 - nav_text.get_width() // 2, 400))
    
    def draw_game_over(self):
        """Draw the game over screen."""
        # Create a semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))  # Black with transparency
        self.screen.blit(overlay, (0, 0))
        
        # Draw game over text
        gameover_text = self.font_large.render("Training Complete!", True, WHITE)
        self.screen.blit(gameover_text, (SCREEN_WIDTH // 2 - gameover_text.get_width() // 2, 150))
        
        # Draw final score
        score_text = self.font_medium.render(f"Final Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (SCREEN_WIDTH // 2 - score_text.get_width() // 2, 250))
        
        # Draw restart instructions
        restart_text = self.font_medium.render("Press SPACE to restart or ESC to quit", True, WHITE)
        self.screen.blit(restart_text, (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, 350))
    
    def run(self):
        """Main game loop."""
        running = True
        
        while running:
            # Process events
            running = self.handle_events()
            
            # Update game state
            self.update()
            
            # Draw everything
            self.draw()
            
            # Cap the frame rate
            self.clock.tick(FPS)
            
    def cleanup(self):
        """Clean up resources before exiting."""
        if self.camera_enabled and self.cap is not None:
            self.cap.release()
        pygame.quit()
        
def main():
    """Main function to run the training mode."""
    # Create and run the training mode
    training = TrainingMode()
    
    try:
        training.run()
    except Exception as e:
        print(f"Error in training mode: {e}")
        import traceback
        traceback.print_exc()
    finally:
        training.cleanup()
        
if __name__ == "__main__":
    main()