#!/usr/bin/env python
"""
Hand Gesture Car Control System - Troubleshooter

This script helps diagnose and fix common issues with the Hand Gesture Car Control System.
It checks dependencies, camera access, and other potential problems.
"""

import sys
import os
import time
import importlib
import subprocess
import traceback

# Add color support for terminal output
COLORS = {
    'GREEN': '\033[92m',
    'YELLOW': '\033[93m',
    'RED': '\033[91m',
    'RESET': '\033[0m',
    'BOLD': '\033[1m'
}

# For terminals that don't support color
if sys.platform.startswith('win'):
    # Check if Windows terminal supports ANSI colors
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except:
        # If not, disable colors
        for key in COLORS:
            COLORS[key] = ''

def print_colored(message, color):
    """Print a message with the specified color"""
    print(f"{COLORS[color]}{message}{COLORS['RESET']}")

def check_python_version():
    """Check if the Python version is adequate"""
    current_version = sys.version_info
    required_version = (3, 7)
    
    print("Checking Python version...", end=" ")
    
    if current_version >= required_version:
        print_colored(f"OK - Python {current_version.major}.{current_version.minor}.{current_version.micro}", "GREEN")
        return True
    else:
        print_colored(f"ERROR - Python {current_version.major}.{current_version.minor}.{current_version.micro} (Required: 3.7+)", "RED")
        print("Please install Python 3.7 or newer from https://www.python.org/downloads/")
        return False

def check_required_packages():
    """Check if all required packages are installed"""
    required_packages = {
        "pygame": "2.0.0",
        "opencv-python": "4.5.0",
        "mediapipe": "0.8.10",
        "numpy": "1.20.0"
    }
    
    all_packages_installed = True
    print("\nChecking required packages:")
    
    for package, min_version in required_packages.items():
        print(f"  {package} (min {min_version})...", end=" ")
        
        try:
            # Convert package name (e.g., opencv-python to opencv_python)
            module_name = package.replace('-', '_')
            
            # Special case for opencv-python
            if module_name == "opencv_python":
                module_name = "cv2"
                
            # Try to import the package
            module = importlib.import_module(module_name)
            
            # Get the version (packages store it differently)
            if hasattr(module, '__version__'):
                version = module.__version__
            elif hasattr(module, 'version'):
                version = module.version
            elif module_name == "cv2":  # Special case for OpenCV
                version = cv2.__version__
            else:
                version = "unknown"
            
            if version != "unknown":
                print_colored(f"OK - v{version}", "GREEN")
            else:
                print_colored(f"WARNING - version unknown", "YELLOW")
                
        except ImportError:
            print_colored("ERROR - Not installed", "RED")
            print(f"    Please install with: pip install {package}>={min_version}")
            all_packages_installed = False
        except Exception as e:
            print_colored(f"ERROR - {str(e)}", "RED")
            all_packages_installed = False
    
    return all_packages_installed

def check_camera_access():
    """Test camera access"""
    print("\nChecking camera access...")
    
    try:
        import cv2
        
        # Try to open the default camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print_colored("ERROR - Could not open default camera", "RED")
            print("  - Make sure a webcam is connected")
            print("  - Check if another application is using the camera")
            print("  - Try restarting your computer")
            return False
        
        # Try to read a frame
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            print_colored("ERROR - Camera opened but could not read frames", "RED")
            print("  - Make sure your webcam is working properly")
            print("  - Check camera permissions")
            return False
        
        # Get camera info
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Test multiple frames to check FPS
        frames = 0
        start_time = time.time()
        
        print("  Reading 30 frames to check performance...", end=" ")
        while frames < 30 and time.time() - start_time < 5:  # Max 5 seconds
            ret, frame = cap.read()
            if not ret:
                break
            frames += 1
            
        end_time = time.time()
        duration = end_time - start_time
        fps = frames / duration if duration > 0 else 0
        
        # Release the camera
        cap.release()
        
        if frames < 30:
            print_colored(f"WARNING - Only captured {frames} frames", "YELLOW")
        else:
            print_colored("OK", "GREEN")
        
        if fps < 15:
            print_colored(f"  Camera performance: {fps:.1f} FPS (LOW, may affect detection)", "YELLOW")
        else:
            print_colored(f"  Camera performance: {fps:.1f} FPS (Good)", "GREEN")
            
        print(f"  Camera resolution: {width}x{height}")
        
        return True
        
    except Exception as e:
        print_colored(f"ERROR - {str(e)}", "RED")
        traceback.print_exc()
        return False

def check_project_files():
    """Check if all required project files exist"""
    required_files = [
        "main.py",
        "training_mode.py",
        "car_control.py",
        "hand_detector/__init__.py"
    ]
    
    # Check directories first
    required_dirs = ["hand_detector", "game"]
    
    print("\nChecking project structure:")
    
    all_files_found = True
    
    # Check directories
    for directory in required_dirs:
        if os.path.isdir(directory):
            print_colored(f"  Directory '{directory}': Found", "GREEN")
        else:
            print_colored(f"  Directory '{directory}': Missing", "RED")
            all_files_found = False
    
    # Check files
    for file_path in required_files:
        if os.path.isfile(file_path):
            print_colored(f"  File '{file_path}': Found", "GREEN")
        else:
            print_colored(f"  File '{file_path}': Missing", "RED")
            all_files_found = False
    
    return all_files_found

def test_mediapipe():
    """Test if MediaPipe is working properly"""
    print("\nTesting MediaPipe hand detection...")
    
    try:
        import mediapipe as mp
        import cv2
        import numpy as np
        
        # Initialize MediaPipe Hands
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
        # Create a simple test image (black background with white hand shape)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw a simple hand shape
        # This won't be detected as a real hand but tests if MediaPipe runs
        cv2.rectangle(img, (200, 200), (300, 400), (255, 255, 255), -1)
        
        # Process the image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        print_colored("  MediaPipe initialized successfully", "GREEN")
        print(f"  MediaPipe version: {mp.__version__}")
        
        # Clean up
        hands.close()
        return True
        
    except Exception as e:
        print_colored(f"ERROR - {str(e)}", "RED")
        traceback.print_exc()
        return False

def fix_display_surface_quit():
    """Try to fix the 'display Surface quit' error"""
    print("\nAttempting to fix 'display Surface quit' error...")
    
    try:
        # This error usually happens because pygame was not properly initialized
        # or pygame.quit() was called prematurely
        # Let's modify the training_mode.py file
        
        if os.path.exists("training_mode.py"):
            with open("training_mode.py", "r") as f:
                content = f.read()
            
            # Check for common issues
            pygame_init_count = content.count("pygame.init()")
            pygame_quit_count = content.count("pygame.quit()")
            
            print(f"  Found {pygame_init_count} calls to pygame.init()")
            print(f"  Found {pygame_quit_count} calls to pygame.quit()")
            
            if pygame_init_count == 0:
                print_colored("  ERROR - No pygame.init() call found", "RED")
                print("  Please add pygame.init() at the beginning of your setup method")
                return False
            
            if pygame_quit_count > 1:
                print_colored("  WARNING - Multiple pygame.quit() calls may cause issues", "YELLOW")
            
            # Check if pygame.init() is properly guarded
            if "if not pygame.get_init():" not in content:
                print_colored("  WARNING - Missing check for pygame initialization", "YELLOW")
                print("  Consider adding 'if not pygame.get_init(): pygame.init()' to prevent multiple initializations")
            
            print("\nSuggested fix for 'display Surface quit' error:")
            print("1. Make sure pygame.init() is called only once at the beginning")
            print("2. Move pygame.quit() to the cleanup method")
            print("3. Add checks before using pygame.display functions")
            print("4. Use try/except blocks around pygame operations")
            
            # Create a backup file
            backup_file = "training_mode_backup.py"
            with open(backup_file, "w") as f:
                f.write(content)
            
            print(f"\nCreated backup at {backup_file}")
            print_colored("Please download and use the fixed training_mode.py file", "GREEN")
            
            return True
        else:
            print_colored("  File training_mode.py not found", "RED")
            return False
    
    except Exception as e:
        print_colored(f"ERROR - {str(e)}", "RED")
        return False

def fix_numpy_ndarray_error():
    """Fix the 'numpy.ndarray' object has no attribute 'get' error"""
    print("\nAttempting to fix 'numpy.ndarray object has no attribute get' error...")
    
    # This error happens when the controls variable is a numpy array instead of a dictionary
    print("This error typically occurs when the hand detector returns the wrong data type.")
    print("The fix requires adding type checking before accessing attributes.")
    print("\nSuggestion: Add this type check in your code:")
    print_colored("""
# Before accessing 'get' method on controls:
if isinstance(controls, dict):
    current_gesture = controls.get('gesture_name', 'No detection')
else:
    current_gesture = 'Invalid controls format'
""", "GREEN")
    
    print("\nAlso check your hand detector's return values to ensure it returns a dictionary with the expected keys.")
    return True

def main():
    """Main function to run troubleshooting"""
    print_colored("\n== Hand Gesture Car Control System - Troubleshooter ==", "BOLD")
    print("This utility will check for common issues and suggest fixes.\n")
    
    # List of checks to perform
    checks = [
        (check_python_version, "Python version"),
        (check_required_packages, "Required packages"),
        (check_camera_access, "Camera access"),
        (check_project_files, "Project files"),
        (test_mediapipe, "MediaPipe functionality")
    ]
    
    # Run all checks and collect results
    results = {}
    for check_function, check_name in checks:
        results[check_name] = check_function()
    
    # Display summary
    print_colored("\n== Summary ==", "BOLD")
    all_passed = True
    for check_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        color = "GREEN" if passed else "RED"
        print(f"{check_name}: {COLORS[color]}{status}{COLORS['RESET']}")
        all_passed = all_passed and passed
    
    # Show fixes for common errors if needed
    if not all_passed:
        print_colored("\n== Fixing Common Errors ==", "BOLD")
        print("The following sections will help fix commonly seen errors:")
        
        # Display specific fixes
        fix_display_surface_quit()
        fix_numpy_ndarray_error()
        
        print_colored("\n== Next Steps ==", "BOLD")
        print("1. Apply the suggested fixes to your code")
        print("2. Make sure all dependencies are properly installed")
        print("3. Try running the application again")
        print("4. If problems persist, please check the error messages carefully")
    else:
        print_colored("\nAll checks passed! Your system is ready to run the Hand Gesture Car Control System.", "GREEN")
        print("\nTo start the application, run:")
        print_colored("  python main.py", "BOLD")
        
    print("\nFor additional help or to report issues, please refer to the documentation.")
    print("Thank you for using the Hand Gesture Car Control System Troubleshooter!")

if __name__ == "__main__":
    main()