#!/usr/bin/env python
"""
Troubleshooting Utility for Hand Gesture Car Control System

This script helps diagnose and fix common issues with the Hand Gesture Car Control System.
It tests various components and provides clear guidance on resolving problems.
"""

import os
import sys
import time
import importlib
import subprocess
import platform

# Test status constants
TEST_PASS = "PASS"
TEST_FAIL = "FAIL"
TEST_WARN = "WARN"

class SystemCheck:
    """Performs system checks and diagnostics"""
    
    def __init__(self):
        self.results = []
        self.issue_count = 0
        
    def run_all_checks(self):
        """Run all system checks"""
        print("Running system diagnostics...")
        print("=============================")
        
        self.check_python_version()
        self.check_required_packages()
        self.check_camera_access()
        self.check_gpu_availability()
        self.check_directories()
        self.check_file_permissions()
        
        # Display summary
        print("\nDiagnostic Summary:")
        print(f"Total issues found: {self.issue_count}")
        
        if self.issue_count == 0:
            print("All tests passed! The system is ready to run.")
        else:
            print("Please resolve the issues above to ensure proper operation.")
            
        return self.issue_count == 0
    
    def record_result(self, test_name, status, message, fix=None):
        """Record a test result"""
        result = {
            "test": test_name,
            "status": status,
            "message": message,
            "fix": fix
        }
        
        self.results.append(result)
        
        if status != TEST_PASS:
            self.issue_count += 1
        
        # Display the result
        status_color = {
            TEST_PASS: "\033[92m",  # Green
            TEST_FAIL: "\033[91m",  # Red
            TEST_WARN: "\033[93m"   # Yellow
        }.get(status, "")
        
        reset_color = "\033[0m"
        
        print(f"{status_color}[{status}]{reset_color} {test_name}: {message}")
        
        if fix:
            print(f"       Fix: {fix}")
    
    def check_python_version(self):
        """Check if Python version is adequate"""
        current_version = sys.version_info
        required_version = (3, 7)
        
        if current_version >= required_version:
            self.record_result(
                "Python Version",
                TEST_PASS,
                f"Python {current_version.major}.{current_version.minor}.{current_version.micro} (Required: 3.7+)"
            )
        else:
            self.record_result(
                "Python Version",
                TEST_FAIL,
                f"Python {current_version.major}.{current_version.minor}.{current_version.micro} (Required: 3.7+)",
                "Install Python 3.7 or newer from https://www.python.org/downloads/"
            )
    
    def check_required_packages(self):
        """Check if all required packages are installed"""
        required_packages = {
            "pygame": "2.0.0",
            "opencv-python": "4.5.0",
            "mediapipe": "0.8.10",
            "numpy": "1.20.0"
        }
        
        for package, min_version in required_packages.items():
            try:
                # Try to import the package
                module = importlib.import_module(package.replace('-', '_'))
                
                # Get the version
                version = getattr(module, '__version__', 'unknown')
                
                # Check if the version is adequate
                if version == 'unknown' or self._compare_versions(version, min_version) >= 0:
                    self.record_result(
                        f"Package: {package}",
                        TEST_PASS,
                        f"Version {version} installed (Required: {min_version}+)"
                    )
                else:
                    self.record_result(
                        f"Package: {package}",
                        TEST_WARN,
                        f"Version {version} installed (Required: {min_version}+)",
                        f"Upgrade package with: pip install --upgrade {package}>={min_version}"
                    )
                    
            except ImportError:
                self.record_result(
                    f"Package: {package}",
                    TEST_FAIL,
                    "Not installed",
                    f"Install package with: pip install {package}>={min_version}"
                )
    
    def check_camera_access(self):
        """Check if cameras are accessible"""
        try:
            import cv2
            
            # Try to open the default camera
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                # Try to read a frame
                ret, frame = cap.read()
                if ret:
                    self.record_result(
                        "Camera Access",
                        TEST_PASS,
                        "Default camera is accessible"
                    )
                else:
                    self.record_result(
                        "Camera Access",
                        TEST_WARN,
                        "Camera opened but could not read frames",
                        "Try restarting your computer or check if another application is using the camera"
                    )
                cap.release()
            else:
                self.record_result(
                    "Camera Access",
                    TEST_FAIL,
                    "Could not access any camera",
                    "Check if a webcam is connected and not in use by another application"
                )
                
        except Exception as e:
            self.record_result(
                "Camera Access",
                TEST_FAIL,
                f"Error accessing camera: {str(e)}",
                "Ensure OpenCV is properly installed and a webcam is connected"
            )
    
    def check_gpu_availability(self):
        """Check if GPU acceleration is available for MediaPipe"""
        try:
            import cv2
            
            # Check for CUDA support in OpenCV
            cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
            
            if cuda_available:
                self.record_result(
                    "GPU Acceleration",
                    TEST_PASS,
                    "CUDA-enabled GPU is available for acceleration"
                )
            else:
                self.record_result(
                    "GPU Acceleration",
                    TEST_WARN,
                    "No GPU acceleration available",
                    "For better performance, consider using a computer with a CUDA-capable GPU"
                )
                
        except Exception as e:
            self.record_result(
                "GPU Acceleration",
                TEST_WARN,
                f"Could not check GPU availability: {str(e)}",
                "The application will run in CPU mode"
            )
    
    def check_directories(self):
        """Check if all required directories exist"""
        required_dirs = [
            "hand_detector",
            "sounds",
            "assets"
        ]
        
        for directory in required_dirs:
            if os.path.isdir(directory):
                self.record_result(
                    f"Directory: {directory}",
                    TEST_PASS,
                    "Directory exists"
                )
            else:
                self.record_result(
                    f"Directory: {directory}",
                    TEST_FAIL,
                    "Directory not found",
                    f"Create the {directory} directory in the application folder"
                )
    
    def check_file_permissions(self):
        """Check if file permissions are correct"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Check read permissions on current directory
        try:
            os.listdir(current_dir)
            self.record_result(
                "File Permissions",
                TEST_PASS,
                "Application has read permissions"
            )
        except PermissionError:
            self.record_result(
                "File Permissions",
                TEST_FAIL,
                "Cannot read files in application directory",
                "Check file permissions or run the application with admin privileges"
            )
        
        # Check write permissions (try to create a temp file)
        try:
            temp_file = os.path.join(current_dir, "temp_write_test.txt")
            with open(temp_file, 'w') as f:
                f.write("Test")
            os.remove(temp_file)
            self.record_result(
                "Write Permissions",
                TEST_PASS,
                "Application has write permissions"
            )
        except PermissionError:
            self.record_result(
                "Write Permissions",
                TEST_WARN,
                "Cannot write files in application directory",
                "Check file permissions or run the application with admin privileges"
            )
    
    def _compare_versions(self, version1, version2):
        """Compare two version strings"""
        # Split versions into components
        v1_parts = version1.split('.')
        v2_parts = version2.split('.')
        
        # Convert to integers for comparison
        v1 = [int(''.join(filter(str.isdigit, part))) for part in v1_parts]
        v2 = [int(''.join(filter(str.isdigit, part))) for part in v2_parts]
        
        # Ensure both lists are of the same length
        length = max(len(v1), len(v2))
        v1.extend([0] * (length - len(v1)))
        v2.extend([0] * (length - len(v2)))
        
        # Compare components
        for i in range(length):
            if v1[i] > v2[i]:
                return 1
            elif v1[i] < v2[i]:
                return -1
        
        return 0


def check_webcam():
    """Run a standalone webcam test"""
    try:
        import cv2
        print("Testing webcam access...")
        
        # Check available cameras
        available_cameras = []
        for i in range(10):  # Check first 10 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"Camera {i} is available")
                    available_cameras.append(i)
                cap.release()
        
        if not available_cameras:
            print("No cameras detected!")
            return False
        
        # Test the first available camera
        camera_index = available_cameras[0]
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Could not open camera {camera_index}")
            return False
        
        # Create a window
        window_name = "Camera Test"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        print("\nCamera preview starting...")
        print("Press 'q' to exit the test")
        
        frames_read = 0
        test_duration = 5  # seconds
        start_time = time.time()
        
        while time.time() - start_time < test_duration:
            ret, frame = cap.read()
            if not ret:
                break
                
            frames_read += 1
            
            # Display frame counter
            cv2.putText(
                frame,
                f"Frames: {frames_read}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Show the frame
            cv2.imshow(window_name, frame)
            
            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        fps = frames_read / min(time.time() - start_time, test_duration)
        print(f"\nCamera test completed:")
        print(f"- Frames captured: {frames_read}")
        print(f"- Approx. FPS: {fps:.1f}")
        
        if frames_read > 0:
            if fps < 15:
                print("WARNING: Low frame rate detected. This may affect hand detection performance.")
            return True
        else:
            print("ERROR: Could not read frames from camera")
            return False
            
    except Exception as e:
        print(f"Camera test failed: {str(e)}")
        return False


def check_mediapipe():
    """Test if MediaPipe is working properly"""
    try:
        import mediapipe as mp
        import cv2
        import numpy as np
        
        print("Testing MediaPipe hand detection...")
        
        # Initialize MediaPipe Hands
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
        # Create a test image (black background)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw a simple hand shape (just for testing, not realistic)
        # This won't be detected as a real hand but tests if MediaPipe runs
        cv2.rectangle(img, (200, 200), (300, 400), (255, 255, 255), -1)
        
        # Process the image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        # Check if MediaPipe processed without errors
        print("MediaPipe successfully processed the test image")
        print("MediaPipe Hands version:", mp.__version__)
        
        # Clean up
        hands.close()
        return True
        
    except Exception as e:
        print(f"MediaPipe test failed: {str(e)}")
        return False


def main():
    """Main function to run troubleshooting"""
    print("Hand Gesture Car Control System - Troubleshooting Utility")
    print("=======================================================")
    print("This utility will help diagnose and fix common issues.\n")
    
    while True:
        print("\nSelect an option:")
        print("1. Run system diagnostics")
        print("2. Test webcam")
        print("3. Test MediaPipe")
        print("4. Exit")
        
        try:
            choice = int(input("Enter your choice (1-4): "))
            
            if choice == 1:
                checker = SystemCheck()
                checker.run_all_checks()
            elif choice == 2:
                check_webcam()
            elif choice == 3:
                check_mediapipe()
            elif choice == 4:
                print("Exiting troubleshooting utility.")
                break
            else:
                print("Invalid choice. Please enter a number between 1 and 4.")
                
        except ValueError:
            print("Invalid input. Please enter a number.")
        except Exception as e:
            print(f"Error: {str(e)}")
            
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()