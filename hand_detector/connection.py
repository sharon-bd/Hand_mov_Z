"""
Hand gesture to car control connection module.
This ensures that the car receives valid control signals from the hand detector.
"""

import time
import threading
import cv2
import numpy as np
import platform
import subprocess

class HandCarConnectionManager:
    """Manages the connection between hand gesture detection and car control"""
    
    def __init__(self, camera_index=0):
        """
        Initialize the connection manager
        
        Args:
            camera_index: Index of the camera to use
        """
        self.camera_index = camera_index
        self.camera = None
        self.running = False
        self.thread = None
        self.simulation_mode = False  # Flag for simulation mode (no camera)
        
        # Control values
        self.controls = {
            'steering': 0.0,
            'throttle': 0.0,
            'braking': False,
            'boost': False,
            'gesture_name': 'Initializing...',
            'speed': 0.0,
            'direction': 0.0
        }
        
        # Stats and debug info
        self.fps = 0.0
        self.last_frame_time = time.time()
        self.frames_processed = 0
        self.successful_detections = 0
        
        # Visuals
        self.latest_frame = None
        self.data_panel = None
        
        # Import modules only when needed to avoid circular imports
        from hand_detector.gestures import HandGestureDetector
        self.gesture_detector = HandGestureDetector()
        
        print("🔄 Hand-Car Connection Manager initialized")
        
    def _check_camera_availability(self):
        """Check camera availability and print diagnostic information"""
        print("📷 Checking camera availability...")
        
        # Try to get system info
        camera_info = "Camera info not available"
        if platform.system() == 'Windows':
            try:
                # Run dxdiag to get system info
                result = subprocess.run(['dxdiag', '/t', 'cameras'], 
                                       capture_output=True, 
                                       text=True, 
                                       timeout=5)
                camera_info = result.stdout[:300]  # First 300 chars only
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
            
            print(f"System camera info: {camera_info}")
        
        # Try to list available cameras using cv2.VideoCapture
        for i in range(3):  # Check first 3 indices
            try:
                cap = cv2.VideoCapture(i)
                ret = cap.isOpened()
                if ret:
                    print(f"✅ Found camera at index {i}")
                    # Get camera properties
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    print(f"   Resolution: {width}x{height}")
                    
                    # Try to read a frame
                    read_ret, frame = cap.read()
                    if read_ret:
                        print(f"   Successfully read frame from camera {i}")
                    else:
                        print(f"   ⚠️ Could not read frame from camera {i}")
                    
                    cap.release()
                else:
                    print(f"❌ Could not open camera at index {i}")
            except Exception as e:
                print(f"❌ Error accessing camera {i}: {e}")
    
    def start(self):
        """Start the camera capture and processing thread"""
        if self.running:
            print("⚠️ Connection already running")
            return True
        
        # First check camera availability
        self._check_camera_availability()
        
        try:
            # Try to open the camera
            self.camera = cv2.VideoCapture(self.camera_index)
            if not self.camera.isOpened():
                print(f"❌ Error: Could not open camera at index {self.camera_index}")
                
                # Try alternative camera indexes
                camera_found = False
                for alt_index in [0, 1, 2]:
                    if alt_index != self.camera_index:
                        print(f"Trying alternative camera index: {alt_index}")
                        self.camera = cv2.VideoCapture(alt_index)
                        if self.camera.isOpened():
                            self.camera_index = alt_index
                            print(f"✅ Successfully opened camera at index {alt_index}")
                            
                            # Test if we can actually read from the camera
                            ret, test_frame = self.camera.read()
                            if ret and test_frame is not None:
                                print(f"✅ Successfully read test frame from camera")
                                camera_found = True
                                break
                            else:
                                print("⚠️ Camera opened but failed to read test frame")
                                self.camera.release()
                                self.camera = None
                        else:
                            print(f"❌ Failed to open camera at index {alt_index}")
                
                if not camera_found:
                    print("⚠️ No working camera found, switching to simulation mode")
                    self.simulation_mode = True
                    self.start_simulation()
                    return True
            else:
                # Try to read a test frame to ensure the camera is working
                ret, test_frame = self.camera.read()
                if not ret or test_frame is None:
                    print("❌ Camera opened but failed to read frame")
                    print("⚠️ Switching to simulation mode")
                    self.simulation_mode = True
                    self.start_simulation()
                    return True
            
            # If we get here, we have a working camera
            # Set camera properties for better performance
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"✅ Camera initialized: {test_frame.shape}")
            
            # Start the processing thread
            self.running = True
            self.thread = threading.Thread(target=self._process_frames)
            self.thread.daemon = True
            self.thread.start()
            
            print("🎮 Hand-Car connection started with real camera")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing camera: {e}")
            print("⚠️ Switching to simulation mode")
            self.simulation_mode = True
            self.start_simulation()
            return True
    
    def start_simulation(self):
        """Start a simulation thread when there's no camera available"""
        print("🎮 Starting hand gesture simulator")
        
        # Initialize simulated camera frame (gray with text)
        self.latest_frame = np.ones((480, 640, 3), dtype=np.uint8) * 100  # Gray
        cv2.putText(
            self.latest_frame, 
            "SIMULATION MODE - NO CAMERA", 
            (80, 240), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 0, 255), 
            2
        )
        
        # Start the simulation thread
        self.running = True
        self.thread = threading.Thread(target=self._simulate_controls)
        self.thread.daemon = True
        self.thread.start()
        
        print("🎮 Hand-Car connection started in simulation mode")
        return True
    
    def _simulate_controls(self):
        """Simulate hand gesture controls when no camera is available"""
        sim_steering = 0.0
        sim_throttle = 0.3
        steering_direction = 0.02  # How fast the steering changes
        
        while self.running:
            try:
                # Simulate steering that gradually moves left and right
                sim_steering += steering_direction
                if abs(sim_steering) > 0.7:
                    steering_direction *= -1  # Reverse direction
                
                # Update time on the simulated frame
                self.latest_frame = np.ones((480, 640, 3), dtype=np.uint8) * 100  # Gray
                cv2.putText(
                    self.latest_frame, 
                    "SIMULATION MODE - NO CAMERA", 
                    (80, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 0, 255), 
                    2
                )
                
                # Add current time
                timestamp = time.strftime("%H:%M:%S")
                cv2.putText(
                    self.latest_frame, 
                    f"Time: {timestamp}", 
                    (80, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 0, 255), 
                    2
                )
                
                # Add controls info
                cv2.putText(
                    self.latest_frame, 
                    f"Steering: {sim_steering:.2f}  Throttle: {sim_throttle:.2f}", 
                    (80, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 0, 255), 
                    2
                )
                
                # Update controls with simulated values
                self.controls = {
                    'steering': sim_steering,
                    'throttle': sim_throttle,
                    'braking': False,
                    'boost': False,
                    'gesture_name': 'Simulation',
                    'speed': sim_throttle,
                    'direction': sim_steering
                }
                
                # Update data panel
                self._update_data_panel()
                
                # Pause to simulate camera frame rate
                time.sleep(0.033)  # ~30fps
                
                # Calculate FPS metrics
                current_time = time.time()
                dt = current_time - self.last_frame_time
                self.last_frame_time = current_time
                
                if dt > 0:
                    instant_fps = 1.0 / dt
                    self.fps = 0.9 * self.fps + 0.1 * instant_fps if self.fps > 0 else instant_fps
                
                self.frames_processed += 1
                
            except Exception as e:
                print(f"❌ Error in simulation: {e}")
                time.sleep(0.5)  # Avoid spinning on errors
    
    def stop(self):
        """Stop the camera capture and processing"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
            
        if self.camera:
            self.camera.release()
            self.camera = None
            
        print("🛑 Hand-Car connection stopped")
        
    def _process_frames(self):
        """Process frames from the camera in a separate thread"""
        frame_count = 0
        while self.running:
            try:
                # Capture a frame
                ret, frame = self.camera.read()
                
                # Debug info every 30 frames
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"📷 Read frame {frame_count}: success={ret}, frame={'valid' if frame is not None else 'None'}")
                
                if not ret or frame is None:
                    print("⚠️ Warning: Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                # Process the frame with hand detector
                try:
                    controls, processed_frame = self.gesture_detector.detect_gestures(frame)
                    
                    # Explicitly verify the processed frame
                    if processed_frame is None:
                        print("⚠️ Hand detector returned None frame, using original")
                        processed_frame = frame
                    
                    # Save the processed frame
                    self.latest_frame = processed_frame
                except Exception as e:
                    print(f"❌ Error in gesture detection: {e}")
                    # Use the original frame if processing fails
                    self.latest_frame = frame
                
                # Update other variables
                controls, processed_frame = self.gesture_detector.detect_gestures(frame)
                
                # Update our control values
                self.controls = controls
                self.latest_frame = processed_frame
                
                # Create a data panel with stats
                self._update_data_panel()
                
                # Calculate FPS
                current_time = time.time()
                dt = current_time - self.last_frame_time
                self.last_frame_time = current_time
                
                if dt > 0:
                    instant_fps = 1.0 / dt
                    self.fps = 0.9 * self.fps + 0.1 * instant_fps if self.fps > 0 else instant_fps
                
                self.frames_processed += 1
                if 'gesture_name' in controls and controls['gesture_name'] != 'No hand detected':
                    self.successful_detections += 1
                    
            except Exception as e:
                print(f"❌ Error in frame processing: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.5)  # Pause briefly to avoid spinning on errors
                
    def _update_data_panel(self):
        """Create a data panel with stats and control values"""
        try:
            # Create a white panel
            self.data_panel = np.ones((500, 600, 3), dtype=np.uint8) * 255
            
            # Add title
            cv2.putText(
                self.data_panel,
                "Hand Gesture Control - Debug Panel",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2
            )
            
            # Add stats
            stats_text = [
                f"FPS: {self.fps:.1f}",
                f"Frames processed: {self.frames_processed}",
                f"Successful detections: {self.successful_detections}",
                f"Detection rate: {self.successful_detections/max(1, self.frames_processed)*100:.1f}%",
            ]
            
            for i, text in enumerate(stats_text):
                cv2.putText(
                    self.data_panel,
                    text,
                    (20, 80 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1
                )
                
            # Add control values
            cv2.putText(
                self.data_panel,
                "Control Values:",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )
            
            control_text = [
                f"Gesture: {self.controls.get('gesture_name', 'Unknown')}",
                f"Steering: {self.controls.get('steering', 0):.2f}",
                f"Throttle: {self.controls.get('throttle', 0):.2f}",
                f"Braking: {self.controls.get('braking', False)}",
                f"Boost: {self.controls.get('boost', False)}"
            ]
            
            for i, text in enumerate(control_text):
                cv2.putText(
                    self.data_panel,
                    text,
                    (20, 240 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1
                )
                
            # Add help text
            help_text = [
                "Controls:",
                "- Move hand left/right: Steer",
                "- Raise/lower hand: Speed up/down",
                "- Make a fist: Brake",
                "- Thumb up: Boost",
                "- Open palm: Stop"
            ]
            
            cv2.putText(
                self.data_panel,
                "Help",
                (320, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )
            
            for i, text in enumerate(help_text):
                cv2.putText(
                    self.data_panel,
                    text,
                    (320, 240 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1
                )
                
        except Exception as e:
            print(f"Error updating data panel: {e}")
            self.data_panel = None
            
    def get_controls(self):
        """Get the current control values"""
        return self.controls
        
    def get_visuals(self):
        """Get the latest frame and data panel"""
        # Create a copy of the latest frame to avoid thread issues
        if self.latest_frame is not None:
            frame_copy = self.latest_frame.copy()
            
            # Add simulation mode indicator if in simulation mode
            if self.simulation_mode:
                cv2.putText(
                    frame_copy, 
                    "SIMULATION MODE", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 0, 255), 
                    2
                )
            
            # Add timestamp
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(
                frame_copy, 
                f"Time: {timestamp}", 
                (10, frame_copy.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                1
            )
            
            return frame_copy, self.data_panel, self.fps
        else:
            # Return a blank frame if no frame is available
            blank_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Gray
            cv2.putText(
                blank_frame,
                "No Camera Feed Available",
                (120, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            
            return blank_frame, self.data_panel, self.fps
