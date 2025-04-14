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
# Use the simplified hand detector instead of the original one
from simple_detector import SimpleHandGestureDetector

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

class TrainingMode:
    """Training mode for hand gesture car control system."""
    
    def __init__(self, camera_index=0):
        """Initialize the training mode
        
        Args:
            camera_index (int): Index of the camera to use for hand detection
        """
        self.screen = None
        self.clock = None
        self.running = False
        self.font = None
        self.camera_index = int(camera_index) if isinstance(camera_index, (str, float, int)) else 0  # Ensure camera_index is an integer
        self.hand_detector = None
        self.car = None
        self.obstacles = []
        self.score = 0
        self.game_time = 0
        self.start_time = 0
        # Add tracking for hand gesture state
        self.last_gesture = None
        self.gesture_start_time = 0
        self.gesture_duration = 0
        # Control smoothing
        self.steering_history = []
        self.throttle_history = []
        
    def setup(self):
        """Set up the training mode environment"""
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Hand Gesture Car Control - Training Mode")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        # Initialize hand detector with proper error handling
        try:
            self.hand_detector = SimpleHandGestureDetector(self.camera_index)
            # Test if the detector is working properly
            self.hand_detector.detect()
        except Exception as e:
            print(f"Error initializing hand detector: {e}")
            print("Falling back to keyboard controls")
            self.hand_detector = None
        
        # Create the player's car in the center of the bottom of the screen
        self.car = Car(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 100)
        
        # Generate some static obstacles for training
        self.generate_training_obstacles()
        
        self.running = True
        self.start_time = time.time()
    
    def generate_training_obstacles(self):
        """Generate static obstacles for the training mode"""
        # Clear any existing obstacles
        self.obstacles = []
        
        # Create a pattern of obstacles for training
        # This is a simplified version - you can make it more elaborate
        for i in range(5):
            obstacle_type = random.choice(OBSTACLE_TYPES)
            x_pos = random.randint(100, SCREEN_WIDTH - 100)
            y_pos = 100 + i * 80  # Spaced vertically
            
            obstacle = {
                "x": x_pos,
                "y": y_pos,
                "width": obstacle_type["width"],
                "height": obstacle_type["height"],
                "color": obstacle_type["color"],
                "type": obstacle_type["name"]
            }
            
            self.obstacles.append(obstacle)
    
    def start(self):
        """Start method for compatibility - just calls the run method"""
        print("Starting Training Mode...")
        try:
            self.run()
        except Exception as e:
            print(f"Error in training mode: {e}")
            self.cleanup()
    
    def get_available_cameras(self):
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
    
    def run(self):
        """Main game loop for training mode"""
        self.setup()
        
        while self.running:
            # Calculate time delta
            dt = self.clock.tick(FPS) / 1000.0
            self.game_time = time.time() - self.start_time
            
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
            
            # Get hand gesture controls
            controls = self.get_controls()
            
            # Update game state
            self.update(controls, dt)
            
            # Draw everything
            self.draw()
            
            # Update the display
            pygame.display.flip()
        
        self.cleanup()
    
    def get_controls(self):
        """Get control inputs from hand gestures or keyboard for testing"""
        controls = {
            'steering': 0,
            'throttle': 0.5,  # Default to medium throttle in training mode
            'braking': False,
            'boost': False,
            'emergency_stop': False
        }
        
        # If hand detector is available, use it
        if self.hand_detector:
            try:
                hand_data = self.hand_detector.detect()
                if hand_data:
                    # Track current gesture and its duration
                    current_gesture = hand_data.get('gesture', None)
                    
                    if current_gesture != self.last_gesture:
                        self.gesture_start_time = time.time()
                        self.last_gesture = current_gesture
                    else:
                        self.gesture_duration = time.time() - self.gesture_start_time
                    
                    # Convert hand position to steering (-1 to 1)
                    normalized_x = hand_data.get('normalized_position_x', 0.5)
                    # Add to steering history for smoothing
                    self.steering_history.append((normalized_x - 0.5) * 2)  # Convert 0-1 to -1 to 1
                    if len(self.steering_history) > 5:  # Keep last 5 samples
                        self.steering_history.pop(0)
                    
                    # Apply smoothing to steering
                    controls['steering'] = sum(self.steering_history) / len(self.steering_history)
                    
                    # Get hand height for throttle control
                    normalized_y = hand_data.get('normalized_position_y', 0.5)
                    # Lower hand position (higher y value) means more throttle
                    throttle_value = 1.0 - normalized_y  # Invert so lower hand = more throttle
                    
                    # Add to throttle history for smoothing
                    self.throttle_history.append(throttle_value)
                    if len(self.throttle_history) > 5:
                        self.throttle_history.pop(0)
                    
                    # Use hand gestures for control mapping as per SRS
                    gesture = current_gesture
                    if gesture == 'fist':
                        controls['braking'] = True
                        controls['throttle'] = 0  # Ensure no throttle when braking
                    elif gesture == 'open_palm':
                        controls['emergency_stop'] = True
                    elif gesture == 'thumb_up':
                        controls['boost'] = True
                    elif gesture == 'pointing':
                        # Special control - can be customized
                        pass
                    else:  # 'steering' or other gestures
                        # Use smoothed throttle for normal driving
                        controls['throttle'] = sum(self.throttle_history) / len(self.throttle_history)
            except Exception as e:
                print(f"Error getting hand controls: {e}")
                # Continue execution with keyboard controls
        
        # Allow keyboard override for testing
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            controls['steering'] = -1
        elif keys[pygame.K_RIGHT]:
            controls['steering'] = 1
        
        if keys[pygame.K_UP]:
            controls['throttle'] = 1.0
        elif keys[pygame.K_DOWN]:
            controls['braking'] = True
        
        if keys[pygame.K_SPACE]:
            controls['boost'] = True
        
        return controls
    
    def update(self, controls, dt):
        """Update game state based on controls and time delta"""
        # Handle emergency stop
        if controls.get('emergency_stop', False):
            self.car.speed = 0
            self.car.direction = 0
            return
            
        # Update car position
        self.car.update(controls, dt)
        
        # Check for collisions with obstacles
        for obstacle in self.obstacles:
            if self.car.check_collision(obstacle):
                # In training mode, just highlight the collision, don't end game
                obstacle['hit'] = True
                # Could add visual feedback or reduce score
            else:
                obstacle['hit'] = False
    
    def draw(self):
        """Draw the game scene"""
        # Fill the background
        self.screen.fill(DARK_GRAY)
        
        # Draw the road
        road_width = 500
        pygame.draw.rect(self.screen, ROAD_COLOR, (
            SCREEN_WIDTH//2 - road_width//2,
            0,
            road_width,
            SCREEN_HEIGHT
        ))
        
        # Draw lane markings
        lane_width = 10
        lane_length = 50
        lane_gap = 20
        lanes = 3  # number of lane dividers
        lane_spacing = road_width / (lanes + 1)
        
        for lane in range(lanes):  # Fixed missing closing parenthesis here
            x_pos = SCREEN_WIDTH//2 - road_width//2 + lane_spacing * (lane + 1)
            for y_pos in range(0, SCREEN_HEIGHT, lane_length + lane_gap):
                pygame.draw.rect(self.screen, LANE_COLOR, (
                    x_pos - lane_width//2,
                    y_pos,
                    lane_width,
                    lane_length
                ))
        
        # Draw obstacles
        for obstacle in self.obstacles:
            color = obstacle['color'] if not obstacle.get('hit', False) else RED
            pygame.draw.rect(self.screen, color, (
                obstacle['x'] - obstacle['width']//2,
                obstacle['y'] - obstacle['height']//2,
                obstacle['width'],
                obstacle['height']
            ))
        
        # Draw the car
        self.car.draw(self.screen)
        
        # Draw HUD information
        self.draw_hud()
    
    def draw_hud(self):
        """Draw heads-up display information"""
        # Training mode title
        title_text = self.font.render("TRAINING MODE", True, WHITE)
        self.screen.blit(title_text, (20, 20))
        
        # Timer
        time_text = self.font.render(f"Time: {int(self.game_time)}s", True, WHITE)
        self.screen.blit(time_text, (20, 60))
        
        # Current gesture
        if hasattr(self, 'last_gesture') and self.last_gesture:
            gesture_text = self.font.render(f"Gesture: {self.last_gesture}", True, WHITE)
            self.screen.blit(gesture_text, (20, 100))
        
        # Instructions
        instructions = [
            "Use hand gestures to control the car:",
            "- Move hand left/right to steer",
            "- Move hand up/down for throttle",
            "- Make fist to brake",
            "- Open palm for emergency stop",
            "- Thumb up for boost",
            "- Point finger for special action",
            "",
            "Press ESC to exit training"
        ]
        
        small_font = pygame.font.Font(None, 24)
        y_pos = SCREEN_HEIGHT - 220
        for line in instructions:
            instr_text = small_font.render(line, True, WHITE)
            self.screen.blit(instr_text, (20, y_pos))
            y_pos += 25
    
    def cleanup(self):
        """Clean up resources before exiting"""
        if self.hand_detector:
            self.hand_detector.release()
        pygame.quit()

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
