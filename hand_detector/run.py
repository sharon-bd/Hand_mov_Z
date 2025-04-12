#!/usr/bin/env python
"""
Run Script for Hand Gesture Car Control System

This script provides a simple way to run the Hand Gesture Car Control System.
It checks for dependencies before launching the main application.
"""

import os
import sys
import subprocess
import platform

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = ['pygame', 'mediapipe', 'opencv-python', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies(packages):
    """Install missing dependencies"""
    print("Installing missing dependencies...")
    
    # Use Python executable from the current environment
    python = sys.executable

    try:
        subprocess.check_call([python, '-m', 'pip', 'install'] + packages)
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("Failed to install dependencies. Please install them manually:")
        print("pip install " + " ".join(packages))
        return False

def find_main_script(start_dir):
    """Find main.py by searching in the start directory and parent directories"""
    print(f"Looking for main.py starting from: {start_dir}")
    
    # First check in the same directory
    main_script = os.path.join(start_dir, 'main.py')
    if os.path.isfile(main_script):
        print(f"Found main.py in the same directory")
        return main_script
        
    # Check in the parent directory
    parent_dir = os.path.dirname(start_dir)
    main_script = os.path.join(parent_dir, 'main.py')
    if os.path.isfile(main_script):
        print(f"Found main.py in parent directory: {parent_dir}")
        return main_script
    
    # Check if main.py is in the project root directory, not hand_detector subfolder
    project_dir = os.path.dirname(os.path.dirname(start_dir))
    main_script = os.path.join(project_dir, 'main.py')
    if os.path.isfile(main_script):
        print(f"Found main.py in project root directory: {project_dir}")
        return main_script
    
    # Check if main.py is in a subdirectory of start_dir
    print("Searching in subdirectories...")
    try:
        for root, _, files in os.walk(start_dir):
            if 'main.py' in files:
                found_path = os.path.join(root, 'main.py')
                print(f"Found main.py in subdirectory: {found_path}")
                return found_path
    except Exception as e:
        print(f"Error while searching subdirectories: {e}")
    
    # Check the current working directory (where the script is being run from)
    cwd = os.getcwd()
    main_script = os.path.join(cwd, 'main.py')
    if os.path.isfile(main_script):
        print(f"Found main.py in current working directory: {cwd}")
        return main_script
        
    print("Could not find main.py in any directory")
    return None

def validate_path(file_path):
    """Validate that a file path is correct and accessible"""
    if not file_path:
        return False
    
    try:
        # Check if path exists
        exists = os.path.isfile(file_path)
        # Check if path is accessible by trying to open it
        if exists:
            with open(file_path, 'r') as f:
                pass  # Just testing if we can open it
        return exists
    except Exception as e:
        print(f"Path validation error for {file_path}: {e}")
        return False

def create_minimal_main_file(location):
    """Create a minimal main.py file if one doesn't exist"""
    print(f"Attempting to create a minimal main.py file at {location}")
    try:
        with open(location, 'w') as f:
            f.write('''#!/usr/bin/env python
"""
Hand Gesture Car Control System Main Application
"""
print("Hand Gesture Car Control System started")
print("This is a minimal placeholder. Please implement your hand detection logic here.")

def main():
    print("Main function running")
    # Your application code would go here
    
if __name__ == "__main__":
    main()
''')
        print("Created a minimal main.py file. Please update it with your application code.")
        return True
    except Exception as e:
        print(f"Failed to create main.py: {e}")
        return False

def main():
    """Main function to run the application"""
    print("Hand Gesture Car Control System Launcher")
    print("========================================")
    
    # Check for Python version
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required.")
        return
    
    # Check for dependencies
    missing_packages = check_dependencies()
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        user_input = input("Do you want to install them now? (y/n): ")
        
        if user_input.lower() in ['y', 'yes']:
            if not install_dependencies(missing_packages):
                return
        else:
            print("Cannot continue without required dependencies.")
            return
    
    # Run the main application
    try:
        print("Starting the application...")
        
        # Get the path to main.py using more robust method
        script_dir = os.path.dirname(os.path.abspath(__file__))
        main_script = find_main_script(script_dir)
        
        if not main_script or not validate_path(main_script):
            print("No valid main.py file found.")
            # Try to create one in the current directory
            cwd = os.getcwd()
            main_script = os.path.join(cwd, 'main.py')
            
            user_choice = input(f"Would you like to create a minimal main.py file in {cwd}? (y/n): ")
            if user_choice.lower() in ['y', 'yes']:
                if create_minimal_main_file(main_script):
                    print(f"Created main.py at {main_script}")
                else:
                    raise FileNotFoundError("Could not find or create main.py")
            else:
                print("Cannot continue without main.py")
                return
        
        print(f"Found main script at: {main_script}")
        # Use absolute path to avoid any path issues
        main_script = os.path.abspath(main_script)
        print(f"Using absolute path: {main_script}")
        
        # Print command we're about to run to help debugging
        command = [sys.executable, main_script]
        print(f"Running command: {' '.join(command)}")
        
        # Handle any potential encoding issues
        if platform.system() == 'Windows':
            subprocess.run(command, check=True, encoding='utf-8')
        else:
            subprocess.run(command, check=True)
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure main.py exists in the project directory.")
        print("Current structure:")
        list_dirs = [
            os.getcwd(),
            script_dir,
            os.path.dirname(script_dir)
        ]
        for directory in list_dirs:
            if os.path.exists(directory):
                print(f"Files in {directory}:")
                try:
                    files = os.listdir(directory)
                    if files:
                        print(", ".join(files))
                    else:
                        print("(empty directory)")
                except Exception as list_err:
                    print(f"Could not list directory: {list_err}")
    except Exception as e:
        print(f"Error launching the application: {e}")
        print(f"Error type: {type(e).__name__}")
        if hasattr(e, 'filename'):
            print(f"Related filename: {e.filename}")
        if hasattr(e, 'stderr'):
            print(f"Error details: {e.stderr}")

if __name__ == "__main__":
    main()