#!/usr/bin/env python
"""
Training Mode for Car Control with Hand Gestures - Optimized Version

This optimized version addresses:
1. Performance issues - reducing OpenCV calls and improving response time
2. Better feedback display - separate resizable windows for data visualization
3. Display options for better customization
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
import threading

# Use the enhanced detector
try:
    from hand_detector.improved_hand_gesture_detector_fixed import EnhancedHandGestureDetector as HandDetector
except ImportError:
    # Fallback
    try:
        from simple_detector import SimpleHandGestureDetector as HandDetector
        print("Warning: Using simple detector instead of enhanced detector.")
    except ImportError:
        print("Error: Could not load any hand gesture detector.")
        HandDetector = None

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


class CameraThread(threading.Thread):
    """Thread class for camera processing to improve performance"""
    
    def __init__(self, camera_index=0):
        """Initialize camera thread"""
        threading.Thread.__init__(self)
        self.camera_index = camera_index
        self.running = False
        self.daemon = True  # Thread will close when main program exits
        
        # Shared data (protected by locks)
        self.lock = threading.Lock()
        self.detector = None
        self.controls = {
            'steering': 0,
            'throttle': 0.5,
            'braking': False,
            'boost': False,
            'gesture_name': 'No detection'
        }
        self.processed_frame = None
        self.data_panel = None
        self.fps = 0
        
        # For FPS calculation
        self.frame_count = 0
        self.start_time = 0
        
    def set_detector(self, detector):
        """Set the hand gesture detector"""
        with self.lock:
            self.detector = detector
            
    def get_controls(self):
        """Get the latest controls from detection thread"""
        with self.lock:
            return self.controls.copy()
            
    def get_visuals(self):
        """Get the visual elements (processed frame, data panel)"""
        with self.lock:
            if self.processed_frame is not None:
                frame_copy = self.processed_frame.copy()
            else:
                frame_copy = None
                
            if self.data_panel is not None:
                panel_copy = self.data_panel.copy()
            else:
                panel_copy = None
                
        return frame_copy, panel_copy, self.fps
            
    def run(self):
        """Main thread function - processes camera input"""
        print("Starting camera processing thread...")
        self.running = True
        self.start_time = time.time()
        self.frame_count = 0
        
        # Initialize camera
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            self.running = False
            return
            
        while self.running:
            # Read camera frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from camera")
                time.sleep(0.1)  # Short delay before trying again
                continue
                
            # Process frame with detector
            with self.lock:
                detector = self.detector
                
            if detector is not None:
                try:
                    # Process frame with hand detector
                    detected_controls, processed_frame, data_panel = detector.detect_gestures(frame)
                    
                    # Update the shared data with lock protection
                    with self.lock:
                        self.controls = detected_controls
                        self.processed_frame = processed_frame
                        self.data_panel = data_panel
                        
                        # Update FPS calculation
                        self.frame_count += 1
                        elapsed_time = time.time() - self.start_time
                        if elapsed_time >= 1.0:  # Update FPS every second
                            self.fps = self.frame_count / elapsed_time
                            self.frame_count = 0
                            self.start_time = time.time()
                            
                except Exception as e:
                    print(f"Error in hand detection: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Add small delay to prevent thread from consuming too much CPU
            time.sleep(0.01)
            
        # Clean up
        cap.release()
        print("Camera thread stopped.")
        
    def stop(self):
        """Stop the camera thread"""
        self.running = False
        self.join(timeout=1.0)  # Wait for thread to finish


class TrainingMode:
    """Training mode for hand gesture car control system."""
    
    def __init__(self, camera_index=0):
        """Initialize the training mode"""
        self.screen = None
        self.clock = None
        self.running = False
        self.font = None
        self.camera_index = int(camera_index) if isinstance(camera_index, (str, float, int)) else 0
        self.hand_detector = None
        self.car = None
        self.obstacles = []
        self.score = 0
        self.game_time = 0
        self.start_time = 0
        
        # Camera thread for better performance
        self.camera_thread = None
        
        # Gesture tracking
        self.last_gesture = None
        self.gesture_start_time = 0
        self.gesture_duration = 0
        
        # Control smoothing
        self.steering_history = []
        self.throttle_history = []
        
        # Display options
        self.show_data_panel = True
        self.show_camera_feed = True
        self.camera_surface = None
        self.data_panel_cv_window = "Hand Gesture Data"
        
        # Performance tracking
        self.fps_game = 0
        self.fps_camera = 0
        self.last_frame_time = 0
        
    def setup(self):
        """Set up the training mode environment"""
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Hand Gesture Car Control - Training Mode")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        # Initialize the camera thread first
        self.camera_thread = CameraThread(self.camera_index)
        
        # Initialize hand detector
        try:
            if HandDetector is not None:
                self.hand_detector = HandDetector()
                print("Initializing hand gesture detector...")
                
                # Pass detector to camera thread
                self.camera_thread.set_detector(self.hand_detector)
                
                # Start camera thread
                self.camera_thread.start()
                
                # Setup separate windows for data display
                if self.show_data_panel:
                    try:
                        cv2.namedWindow(self.data_panel_cv_window, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(self.data_panel_cv_window, 600, 500)  # Larger initial size
                        cv2.moveWindow(self.data_panel_cv_window, SCREEN_WIDTH + 20, 50)  # Position to the right of game window
                    except Exception as e:
                        print(f"Error creating data panel window: {e}")
                        self.show_data_panel = False
            else:
                print("Warning: Hand gesture detector not available. Using keyboard control only.")
        except Exception as e:
            print(f"Error initializing hand detector: {e}")
            print("Using keyboard controls as fallback")
            self.hand_detector = None
        
        # Create the player's car
        self.car = Car(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 100)
        
        # Generate obstacles for training
        self.generate_training_obstacles()
        
        # Initialize empty camera surface
        self.camera_surface = pygame.Surface((320, 240))
        
        self.running = True
        self.start_time = time.time()
        self.last_frame_time = time.time()
    
    def generate_training_obstacles(self):
        """Generate static obstacles for the training mode"""
        self.obstacles = []
        
        for i in range(5):
            obstacle_type = random.choice(OBSTACLE_TYPES)
            x_pos = random.randint(100, SCREEN_WIDTH - 100)
            y_pos = 100 + i * 80
            
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
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def run(self):
        """Main game loop for training mode"""
        self.setup()
        
        while self.running:
            # Calculate time delta and FPS
            current_time = time.time()
            dt = current_time - self.last_frame_time
            self.last_frame_time = current_time
            
            # Safety check to prevent huge dt values
            if dt > 0.1:
                dt = 0.1
                
            # Calculate FPS - smoothed for display
            if dt > 0:
                instantaneous_fps = 1.0 / dt
                # Smooth FPS calculation
                self.fps_game = 0.9 * self.fps_game + 0.1 * instantaneous_fps if self.fps_game > 0 else instantaneous_fps
            
            # Update game time
            self.game_time = time.time() - self.start_time
            
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    # Toggle data panel
                    elif event.key == pygame.K_d:
                        self.show_data_panel = not self.show_data_panel
                        # Create or destroy window as needed
                        if self.show_data_panel:
                            try:
                                cv2.namedWindow(self.data_panel_cv_window, cv2.WINDOW_NORMAL)
                                cv2.resizeWindow(self.data_panel_cv_window, 600, 500)
                            except Exception as e:
                                print(f"Error creating data panel window: {e}")
                        else:
                            try:
                                cv2.destroyWindow(self.data_panel_cv_window)
                            except:
                                pass
                    # Toggle camera feed
                    elif event.key == pygame.K_c:
                        self.show_camera_feed = not self.show_camera_feed
                    # Regenerate obstacles (for practice variety)
                    elif event.key == pygame.K_r:
                        self.generate_training_obstacles()
            
            # Get controls from camera thread
            if self.camera_thread is not None:
                controls = self.camera_thread.get_controls()
                
                # Get camera visuals
                frame, data_panel, self.fps_camera = self.camera_thread.get_visuals()
                
                # Convert camera frame to Pygame surface
                if frame is not None and self.show_camera_feed:
                    try:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_rgb = cv2.resize(frame_rgb, (320, 240))
                        self.camera_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                    except Exception as e:
                        print(f"Error converting camera frame: {e}")
                
                # Update data panel window if needed
                if data_panel is not None and self.show_data_panel:
                    try:
                        # Add game FPS to data panel
                        cv2.putText(
                            data_panel,
                            f"Game FPS: {self.fps_game:.1f}",
                            (20, 470),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 0),
                            1
                        )
                        
                        cv2.imshow(self.data_panel_cv_window, data_panel)
                        cv2.waitKey(1)  # Required to update OpenCV window
                    except Exception as e:
                        print(f"Error showing data panel: {e}")
                
                # Track last gesture and duration
                current_gesture = controls.get('gesture_name', 'No detection')
                if current_gesture != self.last_gesture:
                    self.gesture_start_time = time.time()
                    self.last_gesture = current_gesture
                else:
                    self.gesture_duration = time.time() - self.gesture_start_time
            else:
                # Default controls if no camera thread
                controls = {
                    'steering': 0,
                    'throttle': 0.5,
                    'braking': False,
                    'boost': False,
                    'gesture_name': 'No camera thread'
                }
            
            # Allow keyboard override for testing
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                controls['steering'] = -1
                controls['gesture_name'] = 'Keyboard: Left'
            elif keys[pygame.K_RIGHT]:
                controls['steering'] = 1
                controls['gesture_name'] = 'Keyboard: Right'
            
            if keys[pygame.K_UP]:
                controls['throttle'] = 1.0
                controls['gesture_name'] = 'Keyboard: Accelerate'
            elif keys[pygame.K_DOWN]:
                controls['braking'] = True
                controls['gesture_name'] = 'Keyboard: Brake'
            
            if keys[pygame.K_SPACE]:
                controls['boost'] = True
                controls['gesture_name'] = 'Keyboard: Boost'
            
            # Update game state
            self.update(controls, dt)
            
            # Draw everything
            self.draw(controls)
            
            # Update the display
            pygame.display.flip()
            
            # Cap frame rate
            self.clock.tick(FPS)
    
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
                # In training mode, just highlight the collision
                obstacle['hit'] = True
            else:
                obstacle['hit'] = False
    
    def draw(self, controls):
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
        
        for lane in range(lanes):
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
        
        # Draw camera feed in top-right corner if enabled
        if self.show_camera_feed and self.camera_surface is not None:
            self.screen.blit(self.camera_surface, (SCREEN_WIDTH - 330, 10))
            # Draw border around camera feed
            pygame.draw.rect(self.screen, WHITE, 
                            (SCREEN_WIDTH - 330, 10, 320, 240), 2)
        
        # Draw HUD information
        self.draw_hud(controls)
    
    def draw_hud(self, controls):
        """Draw heads-up display information"""
        # Training mode title
        title_text = self.font.render("Training Mode", True, WHITE)
        self.screen.blit(title_text, (20, 20))
        
        # Performance info
        fps_text = self.font.render(f"FPS: {self.fps_game:.1f} | Camera: {self.fps_camera:.1f}", True, WHITE)
        self.screen.blit(fps_text, (20, 60))
        
        # Current gesture
        gesture = controls.get('gesture_name', 'No detection')
        gesture_text = self.font.render(f"Gesture: {gesture}", True, WHITE)
        self.screen.blit(gesture_text, (20, 100))
        
        # Data panel status
        panel_text = self.font.render(f"Data Display: {'ON' if self.show_data_panel else 'OFF'} (D to toggle)", True, WHITE)
        self.screen.blit(panel_text, (20, 140))
        
        # Camera feed status
        camera_text = self.font.render(f"Camera Feed: {'ON' if self.show_camera_feed else 'OFF'} (C to toggle)", True, WHITE)
        self.screen.blit(camera_text, (20, 180))
        
        # Instructions
        instructions = [
            "Use hand gestures to control the car:",
            "- Move hand left/right to steer",
            "- Move hand up/down to control speed",
            "- Make a fist to brake",
            "- Open palm for emergency stop",
            "- Thumb up for boost",
            "",
            "Keyboard controls:",
            "- Arrow keys for steering and speed",
            "- Space for boost",
            "- D to toggle data display",
            "- C to toggle camera feed",
            "- R to regenerate obstacles",
            "- ESC to exit"
        ]
        
        small_font = pygame.font.Font(None, 24)
        y_pos = SCREEN_HEIGHT - 290
        for line in instructions:
            instr_text = small_font.render(line, True, WHITE)
            self.screen.blit(instr_text, (20, y_pos))
            y_pos += 25
    
    def cleanup(self):
        """Clean up resources before exiting"""
        print("Cleaning up resources...")
        
        # Stop camera thread
        if self.camera_thread is not None:
            self.camera_thread.stop()
        
        # Clean up hand detector if needed
        if self.hand_detector and hasattr(self.hand_detector, 'release'):
            try:
                self.hand_detector.release()
            except Exception as e:
                print(f"Error releasing hand detector: {e}")
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        
        # Quit pygame
        pygame.quit()
        
        print("Cleanup complete.")


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
            steer_text = font.render(f"Steering: {self.direction:.2f}", True, WHITE)
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


# If run directly, start the training mode
if __name__ == "__main__":
    training = TrainingMode()
    training.start()