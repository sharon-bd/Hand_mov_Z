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
        self.target_speed = float(controls.get('throttle', 0))  # וודא שערך זה מגיע כראוי
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
        
        # Update car position - horizontal movement based on steering
        self.x += turn_factor * 10  # Amplified horizontal movement for better control practice
        
        # יישום תנועה ורטיקלית בהתבסס על ה-throttle
        vertical_move_amount = self.max_speed * self.speed * dt
        self.y -= vertical_move_amount  # תנועה כלפי מעלה במסך כאשר ה-throttle גבוה
        
        # Keep car within screen bounds
        self.x = max(self.width//2, min(SCREEN_WIDTH - self.width//2, self.x))
        self.y = max(self.height//2, min(SCREEN_HEIGHT - self.height//2, self.y))
        
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

class TrainingMode:
    """
    Training mode for the car control game to help players practice gestures.
    
    This mode provides a simplified environment where players can:
    - Control a car with hand gestures
    - Practice avoiding static obstacles
    - Get visual feedback on control inputs
    - Learn the control system without time pressure
    """
    
    def __init__(self, screen):
        """Initialize training mode."""
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.running = False
        
        # Initialize the hand detector
        self.detector = SimpleHandGestureDetector()
        self.camera = None
        self.camera_index = 0
        
        # Game objects
        self.car = Car(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        self.obstacles = []
        
        # Create some static obstacles for practice
        for i in range(5):
            obstacle_type = random.choice(OBSTACLE_TYPES)
            x = random.randint(100, SCREEN_WIDTH - 100)
            y = random.randint(100, SCREEN_HEIGHT - 100)
            
            # Ensure obstacles aren't too close to starting position
            while math.sqrt((x - self.car.x)**2 + (y - self.car.y)**2) < 150:
                x = random.randint(100, SCREEN_WIDTH - 100)
                y = random.randint(100, SCREEN_HEIGHT - 100)
                
            self.obstacles.append({
                "x": x,
                "y": y,
                "width": obstacle_type["width"],
                "height": obstacle_type["height"],
                "color": obstacle_type["color"],
                "type": obstacle_type["name"]
            })
        
        # Training mode UI
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.instruction_visible = True
        self.show_debug_info = True
        
        # Camera feed display
        self.camera_feed = None
        self.control_panel = None

    def start(self):
        """Start the training mode."""
        self.running = True
        
        # Initialize camera
        self._setup_camera()
        
        # Run the game loop
        self.run()
        
    def run(self):
        """Main game loop for training mode."""
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0  # Delta time in seconds
            
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_h:
                        # Toggle instruction visibility
                        self.instruction_visible = not self.instruction_visible
                    elif event.key == pygame.K_d:
                        # Toggle debug info
                        self.show_debug_info = not self.show_debug_info
            
            # Process hand controls
            controls = self._process_camera_input()
            
            # Update game objects
            self.car.update(controls, dt)
            
            # Draw everything
            self._draw()
            
            # Cap the frame rate
            pygame.display.flip()
        
        # Clean up
        self._cleanup()
            
    def _setup_camera(self):
        """Initialize the camera for hand detection."""
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            if not self.camera.isOpened():
                print(f"Warning: Could not open camera {self.camera_index}!")
                # Try to find any available camera
                available_cameras = get_available_cameras()
                if available_cameras:
                    self.camera_index = available_cameras[0]["index"]
                    print(f"Attempting to use camera {self.camera_index} instead.")
                    self.camera = cv2.VideoCapture(self.camera_index)
                else:
                    print("No cameras found! Continuing without hand controls.")
        except Exception as e:
            print(f"Error initializing camera: {e}")
            self.camera = None
    
    def _process_camera_input(self):
        """Process camera input for hand detection and controls."""
        default_controls = {
            'steering': 0.0,
            'throttle': 0.0,
            'braking': False,
            'boost': False
        }
        
        if not self.camera:
            return default_controls
            
        try:
            ret, frame = self.camera.read()
            if not ret or frame is None:
                return default_controls
                
            # Process the frame with our hand detector
            controls, processed_frame, control_panel = self.detector.detect_gestures(frame)
            
            # Resize for display
            display_height = 180
            display_width = int(frame.shape[1] * (display_height / frame.shape[0]))
            self.camera_feed = cv2.resize(processed_frame, (display_width, display_height))
            self.camera_feed = cv2.cvtColor(self.camera_feed, cv2.COLOR_BGR2RGB)
            self.camera_feed = pygame.surfarray.make_surface(self.camera_feed.swapaxes(0, 1))
            
            # Resize control panel for display
            control_panel_resized = cv2.resize(control_panel, (display_width, display_height))
            control_panel_resized = cv2.cvtColor(control_panel_resized, cv2.COLOR_BGR2RGB)
            self.control_panel = pygame.surfarray.make_surface(control_panel_resized.swapaxes(0, 1))
            
            return controls
        except Exception as e:
            print(f"Error in camera processing: {e}")
            return default_controls
    
    def _draw(self):
        """Draw all game elements."""
        # Clear the screen
        self.screen.fill(DARK_GRAY)
        
        # Draw a simple road
        road_width = 600
        pygame.draw.rect(self.screen, ROAD_COLOR, 
                        (SCREEN_WIDTH//2 - road_width//2, 0, road_width, SCREEN_HEIGHT))
        
        # Draw lane markers
        lane_width = 10
        lane_height = 40
        lane_gap = 30
        num_lanes = SCREEN_HEIGHT // (lane_height + lane_gap) + 1
        
        for i in range(num_lanes):
            y = i * (lane_height + lane_gap)
            pygame.draw.rect(self.screen, LANE_COLOR,
                            (SCREEN_WIDTH//2 - lane_width//2, y, lane_width, lane_height))
        
        # Draw obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, obstacle["color"],
                            (obstacle["x"] - obstacle["width"]//2,
                             obstacle["y"] - obstacle["height"]//2,
                             obstacle["width"], obstacle["height"]))
            
            # Draw labels for obstacles if debug is enabled
            if self.show_debug_info:
                label = self.small_font.render(obstacle["type"], True, WHITE)
                self.screen.blit(label, 
                                (obstacle["x"] - label.get_width()//2,
                                 obstacle["y"] - obstacle["height"] - 25))
        
        # Draw the car
        self.car.draw(self.screen)
        
        # Draw the camera feed and control panel if available
        if self.camera_feed is not None:
            self.screen.blit(self.camera_feed, (10, 10))
        
        if self.control_panel is not None:
            self.screen.blit(self.control_panel, (10, 200))
        
        # Draw instructions if enabled
        if self.instruction_visible:
            self._draw_instructions()
        
        # Draw mode title
        title = self.font.render("TRAINING MODE", True, WHITE)
        self.screen.blit(title, (SCREEN_WIDTH//2 - title.get_width()//2, 20))
        
        # Draw help message
        help_text = self.small_font.render("Press H to toggle instructions, ESC to exit", True, WHITE)
        self.screen.blit(help_text, (SCREEN_WIDTH//2 - help_text.get_width()//2, SCREEN_HEIGHT - 30))
    
    def _draw_instructions(self):
        """Draw instruction panel with control guide."""
        # Create semi-transparent overlay
        overlay = pygame.Surface((400, 300), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))  # Semi-transparent black
        
        # Position in the bottom right
        overlay_x = SCREEN_WIDTH - 420
        overlay_y = SCREEN_HEIGHT - 320
        self.screen.blit(overlay, (overlay_x, overlay_y))
        
        # Draw heading
        title = self.font.render("Training Instructions", True, WHITE)
        self.screen.blit(title, (overlay_x + 20, overlay_y + 20))
        
        # Draw control instructions
        instructions = [
            "• Move your hand left/right to steer",
            "• Raise your hand to accelerate",
            "• Lower your hand to decelerate",
            "• Make a fist to brake",
            "• Practice avoiding the obstacles",
            "• Watch the indicators for feedback"
        ]
        
        for i, instruction in enumerate(instructions):
            text = self.small_font.render(instruction, True, WHITE)
            self.screen.blit(text, (overlay_x + 30, overlay_y + 70 + i * 30))
    
    def _cleanup(self):
        """Clean up resources when exiting."""
        if self.camera:
            self.camera.release()