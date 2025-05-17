"""
Hand gesture to car control connection module.
This ensures that the car receives valid control signals from the hand detector.
"""

import time
import threading
import cv2
import numpy as np

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
        
    def start(self):
        """Start the camera capture and processing thread"""
        if self.running:
            print("⚠️ Connection already running")
            return
            
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            if not self.camera.isOpened():
                print(f"❌ Error: Could not open camera at index {self.camera_index}")
                # Try alternative camera indexes
                for alt_index in [0, 1, 2]:
                    if alt_index != self.camera_index:
                        print(f"Trying alternative camera index: {alt_index}")
                        self.camera = cv2.VideoCapture(alt_index)
                        if self.camera.isOpened():
                            self.camera_index = alt_index
                            print(f"✅ Successfully opened camera at index {alt_index}")
                            break
                
                if not self.camera.isOpened():
                    print("❌ Failed to open any camera")
                    return False
                
            # Set camera properties for better performance
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Read a test frame to verify camera works
            ret, test_frame = self.camera.read()
            if not ret or test_frame is None:
                print("❌ Camera opened but failed to read frame")
                return False
                
            print(f"✅ Camera initialized: {test_frame.shape}")
            
            # Start the processing thread
            self.running = True
            self.thread = threading.Thread(target=self._process_frames)
            self.thread.daemon = True
            self.thread.start()
            
            print("🎮 Hand-Car connection started")
            return True
        except Exception as e:
            print(f"❌ Error initializing camera: {e}")
            return False
            
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
        consecutive_failures = 0
        while self.running:
            try:
                # Capture a frame
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    print(f"⚠️ Warning: Failed to capture frame ({consecutive_failures})")
                    
                    # If too many consecutive failures, try to reinitialize the camera
                    if consecutive_failures > 5:
                        print("🔄 Attempting to reinitialize camera")
                        self.camera.release()
                        self.camera = cv2.VideoCapture(self.camera_index)
                        
                        if not self.camera.isOpened():
                            print("❌ Failed to reinitialize camera")
                            time.sleep(0.5)
                            continue
                        
                        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        consecutive_failures = 0
                    
                    time.sleep(0.1)
                    continue
                
                # Reset failure counter on successful frame
                consecutive_failures = 0
                
                # Process the frame with hand detector
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
        return self.latest_frame, self.data_panel, self.fps
