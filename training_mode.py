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
from game.car import Car
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
            return self.controls.copy() if isinstance(self.controls, dict) else {
                'steering': 0,
                'throttle': 0.5,
                'braking': False,
                'boost': False,
                'gesture_name': 'No detection'
            }
            
    def get_visuals(self):
        """Get the visual elements (processed frame, data panel)"""
        with self.lock:
            # Check if frames are valid before returning
            if self.processed_frame is not None and isinstance(self.processed_frame, np.ndarray):
                frame_copy = self.processed_frame.copy()
            else:
                # Create a blank frame if none is available
                frame_copy = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    frame_copy,
                    "No camera feed available",
                    (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )
                
            if self.data_panel is not None and isinstance(self.data_panel, np.ndarray):
                panel_copy = self.data_panel.copy()
            else:
                # Create a blank data panel if none is available
                panel_copy = np.ones((500, 600, 3), dtype=np.uint8) * 255
                cv2.putText(
                    panel_copy,
                    "No data panel available",
                    (50, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2
                )
                
        return frame_copy, panel_copy, self.fps
            
    def run(self):
        """Main game loop for camera processing thread"""
        self.running = True
        self.start_time = time.time()
        
        # Initialize camera capture
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            self.running = False
            return
            
        # Main processing loop
        while self.running:
            # Read frame from camera
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Error: Could not read frame")
                continue
                
            # Process frame with detector if available
            if self.detector is not None:
                try:
                    # First, explicitly check if the detector has the required methods
                    has_process_frame = hasattr(self.detector, 'process_frame') and callable(getattr(self.detector, 'process_frame'))
                    has_detect = hasattr(self.detector, 'detect') and callable(getattr(self.detector, 'detect'))
                    
                    if not (has_process_frame or has_detect):
                        # If detector has neither method, log once and create simple implementations
                        if not hasattr(self, '_method_warning_shown'):
                            print("Warning: Detector has neither process_frame nor detect method. Using manual fallback.")
                            self._method_warning_shown = True
                        
                        # Create a processed frame with basic information
                        processed_frame = frame.copy()
                        # Draw a message on the frame
                        cv2.putText(
                            processed_frame,
                            "Detector interface not implemented",
                            (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),  # Red color for warning
                            2
                        )
                        
                        # Create a basic data panel
                        data_panel = np.ones((500, 600, 3), dtype=np.uint8) * 255  # White background
                        cv2.putText(
                            data_panel,
                            "Detector Missing Methods: process_frame and detect",
                            (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),  # Red color for warning
                            2
                        )
                        
                        # Use default controls
                        detected_controls = {
                            'steering': 0,
                            'throttle': 0.5,
                            'braking': False,
                            'boost': False,
                            'gesture_name': 'No detection interface'
                        }
                    elif has_process_frame:
                        # Process the frame and get controls using process_frame
                        processed_frame, data_panel, detected_controls = self.detector.process_frame(frame)
                    elif has_detect:
                        # Try using detect method with fallback implementation
                        result = self.detector.detect(frame)
                        
                        # Create a basic visualization
                        processed_frame = frame.copy()
                        data_panel = np.ones((500, 600, 3), dtype=np.uint8) * 255  # White background
                        
                        # Set default controls
                        detected_controls = {
                            'steering': 0,
                            'throttle': 0.5,
                            'braking': False,
                            'boost': False,
                            'gesture_name': 'Unknown'
                        }
                        
                        # Extract controls from result if available
                        if isinstance(result, dict):
                            if 'gesture' in result:
                                detected_controls['gesture_name'] = result['gesture']
                            
                            if 'steering' in result:
                                detected_controls['steering'] = result['steering']
                            
                            if 'throttle' in result:
                                detected_controls['throttle'] = result['throttle']
                            
                            if 'braking' in result:
                                detected_controls['braking'] = result['braking']
                            
                            if 'boost' in result:
                                detected_controls['boost'] = result['boost']
                        
                        # Draw basic info on data panel
                        cv2.putText(
                            data_panel,
                            f"Gesture: {detected_controls['gesture_name']}",
                            (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 0),
                            2
                        )
                        
                        # Add steering and throttle indicators
                        cv2.putText(
                            data_panel,
                            f"Steering: {detected_controls['steering']:.2f}",
                            (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 0),
                            1
                        )
                        
                        cv2.putText(
                            data_panel,
                            f"Throttle: {detected_controls['throttle']:.2f}",
                            (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 0),
                            1
                        )
                    
                    # Ensure controls is a dictionary
                    if not isinstance(detected_controls, dict):
                        # Check if it's a numpy array and try to convert it
                        if isinstance(detected_controls, np.ndarray):
                            try:
                                # If it's a numpy array with correct shape, extract and normalize values
                                if detected_controls.size >= 5:
                                    array_values = detected_controls.flatten()
                                    # ===== FIXED: Properly normalize values from numpy array =====
                                    detected_controls = {
                                        'steering': float(array_values[0]) / 255.0 * 2.0 - 1.0,  # Normalize to [-1.0, 1.0]
                                        'throttle': float(array_values[1]) / 255.0,  # Normalize to [0.0, 1.0]
                                        'braking': bool(array_values[2]),
                                        'boost': bool(array_values[3]),
                                        'gesture_name': 'From array (normalized)'
                                    }
                                else:
                                    print(f"Warning: detected_controls is a numpy array with insufficient elements: {detected_controls.shape}")
                                    detected_controls = {
                                        'steering': 0.0,
                                        'throttle': 0.0,
                                        'braking': False,
                                        'boost': False,
                                        'gesture_name': 'Invalid array format'
                                    }
                            except Exception as e:
                                print(f"Error converting numpy array to controls: {e}")
                                detected_controls = {
                                    'steering': 0.0,
                                    'throttle': 0.0,
                                    'braking': False,
                                    'boost': False,
                                    'gesture_name': 'Conversion error'
                                }
                        else:
                            # Not a dict or numpy array
                            print(f"Warning: detected_controls is not a dictionary: {type(detected_controls)}")
                            detected_controls = {
                                'steering': 0.0,
                                'throttle': 0.0,
                                'braking': False,
                                'boost': False,
                                'gesture_name': 'Invalid controls format'
                            }
                    
                    # Update controls with thread safety
                    with self.lock:
                        # ===== FIXED: Ensure values are properly normalized =====
                        if isinstance(detected_controls, dict):
                            normalized_controls = detected_controls.copy()
                            try:
                                # Normalize steering if needed
                                steering = float(normalized_controls.get('steering', 0.0))
                                if abs(steering) > 1.0:  # Assume steering is in [0, 255] range
                                    normalized_controls['steering'] = steering / 255.0 * 2.0 - 1.0  # Convert to [-1.0, 1.0]
                                
                                # Normalize throttle if needed
                                throttle = float(normalized_controls.get('throttle', 0.0))
                                if throttle > 1.0:  # Assume throttle is in [0, 255] range
                                    normalized_controls['throttle'] = throttle / 255.0  # Convert to [0.0, 1.0]
                            except Exception as e:
                                print(f"Error normalizing controls: {e}")
                                normalized_controls['steering'] = 0.0
                                normalized_controls['throttle'] = 0.0
                                
                            # Ensure boolean values
                            normalized_controls['braking'] = bool(normalized_controls.get('braking', False))
                            normalized_controls['boost'] = bool(normalized_controls.get('boost', False))
                            
                            self.controls = normalized_controls
                        else:
                            self.controls = {
                                'steering': 0.0,
                                'throttle': 0.0,
                                'braking': False,
                                'boost': False,
                                'gesture_name': 'Invalid format (not a dictionary)'
                            }
                            
                        self.processed_frame = processed_frame
                        self.data_panel = data_panel
                        
                        # Update FPS calculation
                        self.frame_count += 1
                        elapsed_time = time.time() - self.start_time
                        if elapsed_time > 1.0:  # Update FPS every second
                            self.fps = self.frame_count / elapsed_time
                            self.frame_count = 0
                            self.start_time = time.time()
                            
                except Exception as e:
                    print(f"Error in detector processing: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Brief sleep to prevent excessive CPU usage
            time.sleep(0.01)
            
        # Clean up
        cap.release()
        
    def stop(self):
        """Stop the camera thread"""
        self.running = False
        self.join(timeout=1.0)  # Wait for thread to finish


class TrainingMode:
    """Training mode for hand gesture car control system."""
    
    def __init__(self, screen_or_camera=0):
        """Initialize the training mode"""
        # Check if the parameter is a screen or camera index
        if isinstance(screen_or_camera, pygame.Surface):
            self.screen = screen_or_camera
            self.camera_index = 0  # Default camera index
        else:
            self.screen = None
            self.camera_index = int(screen_or_camera) if isinstance(screen_or_camera, (str, float, int)) else 0
        
        self.clock = None
        self.running = False
        self.font = None
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
        self.camera_window_name = None
        
    def setup(self):
        """Set up the training mode environment"""
        pygame.init()
        
        # Create a new screen if one is not provided
        if self.screen is None:
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

                # --- Add separate Hand Camera View window ---
                try:
                    camera_window_name = "Hand Camera View"
                    cv2.namedWindow(camera_window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(camera_window_name, 640, 480)         # initial size
                    cv2.moveWindow(camera_window_name, SCREEN_WIDTH + 20, 550)  # below data panel
                    self.camera_window_name = camera_window_name
                except Exception as e:
                    print(f"Error creating camera window: {e}")
                    self.camera_window_name = None
                # --- end addition ---

            else:
                print("Warning: Hand gesture detector not available. Using keyboard control only.")
        except Exception as e:
            print(f"Error initializing hand detector: {e}")
            print("Using keyboard controls as fallback")
            self.hand_detector = None
        
        # Create the player's car
        self.car = Car(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 100)
        
        # Set screen dimensions for boundary checking
        if hasattr(self.car, 'set_screen_dimensions'):
            self.car.set_screen_dimensions(SCREEN_WIDTH, SCREEN_HEIGHT)
        
        # Generate obstacles for training
        self.generate_training_obstacles()
        
        # Initialize empty camera surface
        self.camera_surface = pygame.Surface((320, 240))
        
        self.running = True
        self.start_time = time.time()
        self.last_frame_time = time.time()
        
        return True  # Indicate successful setup
    
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
        setup_result = self.setup()  # Save the setup result
        
        if not setup_result:  # Exit if setup failed
            print("Setup failed. Exiting training mode.")
            return False
        
        try:
            while self.running:
                # Check if pygame is still initialized and screen is valid
                if not pygame.get_init() or self.screen is None:
                    print("Pygame not initialized or screen is None. Exiting.")
                    return False
                    
                try:
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
                    controls = {}
                    if self.camera_thread is not None:
                        controls = self.camera_thread.get_controls()
                        
                        # Ensure controls is a dictionary
                        if not isinstance(controls, dict):
                            print(f"Warning: controls is not a dictionary: {type(controls)}")
                            controls = {
                                'steering': 0,
                                'throttle': 0.5,
                                'braking': False,
                                'boost': False,
                                'gesture_name': 'Invalid controls format'
                            }
                        
                        # Get camera visuals
                        frame, data_panel, self.fps_camera = self.camera_thread.get_visuals()
                        
                        # Show camera feed in separate OpenCV window if enabled
                        if frame is not None and self.show_camera_feed and self.camera_window_name:
                            try:
                                # Display the original frame in the separate window
                                cv2.imshow(self.camera_window_name, frame)
                                cv2.waitKey(1)  # Required to update OpenCV window
                            except Exception as e:
                                print(f"Error showing camera feed: {e}")

                        # Convert camera frame to Pygame surface (for in-game preview)
                        if frame is not None and self.show_camera_feed:
                            try:
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                frame_rgb = cv2.resize(frame_rgb, (320, 240))
                                # Create surface from array
                                camera_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                                if camera_surface is not None:
                                    self.camera_surface = camera_surface
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
                    else:
                        # Default controls if no camera thread
                        controls = {
                            'steering': 0,
                            'throttle': 0.5,
                            'braking': False,
                            'boost': False,
                            'gesture_name': 'No camera thread'
                        }
                    
                    # Use improved debug configuration
                    try:
                        from debug_config import is_debug_enabled, log, can_log
                        
                        # Only log steering value when it changes or periodically
                        steering_value = controls.get('steering', 0)
                        if can_log('controls', steering_value):
                            log('controls', f"Before keyboard: steering={steering_value}", value=steering_value)
                    except ImportError:
                        # Fallback for old debug_config
                        if hasattr(self, '_last_debug_steering') and self._last_debug_steering == controls.get('steering', 0):
                            pass  # Skip logging if unchanged
                        else:
                            self._last_debug_steering = controls.get('steering', 0)
                            if hasattr(self, '_debug_counter') and self._debug_counter % 20 == 0:  # Only log every 20 frames
                                print(f"DEBUG - Before keyboard: steering={controls.get('steering', 0)}")
                            
                            # Initialize counter if needed
                            if not hasattr(self, '_debug_counter'):
                                self._debug_counter = 0
                            self._debug_counter += 1
                    
                    # Store original gesture info before keyboard override
                    original_gesture = controls.get('gesture_name', 'Unknown')
                    
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
                except Exception as e:
                    print(f"Error in main game loop: {e}")
                    import traceback
                    traceback.print_exc()
                    return False
                    
            return True
        except Exception as e:
            print(f"Error in main game loop: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # Cleanup specific to TrainingMode
            pass  
        
        return True
    
    def update(self, controls, dt):
        """Update game state based on controls and elapsed time"""
        
        # Ensure controls is a dictionary
        if not isinstance(controls, dict):
            print(f"WARNING: controls is not a dictionary, type: {type(controls)}")
            controls = {
                'steering': 0.0,
                'throttle': 0.0,
                'braking': False,
                'boost': False,
                'gesture_name': 'Invalid controls format'
            }
        
        # Normalize control values
        controls['steering'] = float(max(-1.0, min(1.0, controls.get('steering', 0.0))))
        controls['throttle'] = float(max(0.0, min(1.0, controls.get('throttle', 0.0))))
        controls['braking'] = bool(controls.get('braking', False))
        controls['boost'] = bool(controls.get('boost', False))
        
        # Debug log for controls
        print(f"DEBUG: Sending to car - steering: {controls['steering']:.2f}, throttle: {controls['throttle']:.2f}")
        
        # Update the car
        if hasattr(self, 'car') and self.car is not None:
            self.car.update(controls, dt)
        else:
            print("ERROR: Car object is None or not initialized")
    
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

        # Draw HUD information
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
        camera_text = self.font.render(f"Camera Feed: {'ON - in separate window' if self.show_camera_feed else 'OFF'} (C to toggle)", True, WHITE)
        self.screen.blit(camera_text, (20, 180))

    def cleanup(self):
        """ניקוי משאבים לפני יציאה"""
        print("ניקוי משאבים...")

        # עצירת חוט המצלמה
        if self.camera_thread is not None:
            self.camera_thread.stop()

        # שחרור גלאי תנועות היד אם נדרש
        if self.hand_detector and hasattr(self.hand_detector, 'release'):
            try:
                self.hand_detector.release()
            except Exception as e:
                print(f"שגיאה בשחרור גלאי תנועות היד: {e}")

        # סגירת כל חלונות OpenCV
        cv2.destroyAllWindows()

        # הערה: אין לסגור את pygame, כיוון שה-GameLauncher עשוי עדיין להשתמש בו
        print("ניקוי הושלם.")