#!/usr/bin/env python
"""
Hand Gesture Car Control System - Main Entry Point - FIXED VERSION

This is a corrected version of the main entry point that handles common issues
and provides better error handling and debugging information.
"""

import os
import sys
import time
import pygame
import cv2
import traceback

# Add better error handling for imports
def safe_import(module_name, package_name=None):
    """Safely import a module with detailed error reporting"""
    try:
        if package_name:
            module = __import__(module_name)
            return getattr(module, package_name)
        else:
            return __import__(module_name)
    except ImportError as e:
        print(f"❌ Failed to import {module_name}: {e}")
        print(f"💡 Try: pip install {package_name or module_name}")
        return None
    except Exception as e:
        print(f"❌ Error importing {module_name}: {e}")
        return None

class TrainingMode:
    def __init__(self):
        self.running = True
        self.screen = None
        self.clock = None
        self.font = None
        
        # Error tracking
        self.initialization_errors = []
        
    def check_dependencies(self):
        """Check if all required dependencies are available"""
        print("🔍 Checking dependencies...")
        
        dependencies = {
            'pygame': 'pygame',
            'cv2': 'opencv-python', 
            'mediapipe': 'mediapipe',
            'numpy': 'numpy'
        }
        
        missing = []
        for module, package in dependencies.items():
            result = safe_import(module, package if module != package else None)
            if result is None:
                missing.append(package)
            else:
                print(f"✅ {module} - OK")
        
        if missing:
            print(f"❌ Missing dependencies: {', '.join(missing)}")
            print("🛠️ Install them with:")
            print(f"   pip install {' '.join(missing)}")
            return False
        
        print("✅ All dependencies found!")
        return True

    def setup(self):
        """Setup the training mode with proper error handling"""
        print("🚀 Starting Hand Gesture Car Control System...")
        
        # Check dependencies first
        if not self.check_dependencies():
            return False
        
        try:
            # Initialize pygame with error checking
            print("🎮 Initializing pygame...")
            
            if not pygame.get_init():
                pygame.init()
                print("✅ pygame initialized successfully")
            else:
                print("✅ pygame already initialized")
            
            # Check if display is available
            if not pygame.display.get_init():
                pygame.display.init()
                print("✅ pygame display initialized")
            
            # Create screen with error handling
            print("🖥️ Creating game window...")
            try:
                self.screen = pygame.display.set_mode((800, 600))
                pygame.display.set_caption("Hand Gesture Car Control - Training Mode")
                print("✅ Game window created successfully")
            except Exception as e:
                print(f"❌ Failed to create game window: {e}")
                self.initialization_errors.append(f"Display error: {e}")
                return False
            
            # Initialize clock and font
            self.clock = pygame.time.Clock()
            
            try:
                self.font = pygame.font.Font(None, 36)
                print("✅ Font loaded successfully")
            except Exception as e:
                print(f"⚠️ Font loading failed, using default: {e}")
                self.font = pygame.font.SysFont(None, 36)
            
            print("✅ Setup completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            traceback.print_exc()
            self.initialization_errors.append(f"Setup error: {e}")
            return False

    def draw_error_screen(self):
        """Draw an error screen with helpful information"""
        if not self.screen:
            return
            
        self.screen.fill((50, 50, 50))  # Dark gray background
        
        if not self.font:
            return
        
        # Title
        title = self.font.render("Setup Error", True, (255, 100, 100))
        self.screen.blit(title, (50, 50))
        
        # Error messages
        y_pos = 120
        for error in self.initialization_errors:
            # Split long error messages
            if len(error) > 60:
                words = error.split()
                line = ""
                for word in words:
                    if len(line + word) < 60:
                        line += word + " "
                    else:
                        if line:
                            text = pygame.font.Font(None, 24).render(line.strip(), True, (255, 255, 255))
                            self.screen.blit(text, (50, y_pos))
                            y_pos += 30
                        line = word + " "
                if line:
                    text = pygame.font.Font(None, 24).render(line.strip(), True, (255, 255, 255))
                    self.screen.blit(text, (50, y_pos))
                    y_pos += 30
            else:
                text = pygame.font.Font(None, 24).render(error, True, (255, 255, 255))
                self.screen.blit(text, (50, y_pos))
                y_pos += 40
        
        # Instructions
        if y_pos < 500:
            instructions = [
                "Press ESC to exit",
                "Check the console for detailed error information"
            ]
            
            for instruction in instructions:
                text = pygame.font.Font(None, 28).render(instruction, True, (255, 255, 0))
                self.screen.blit(text, (50, y_pos))
                y_pos += 35

    def draw_menu(self):
        """Draw a simple test menu"""
        if not self.screen or not self.font:
            return
            
        self.screen.fill((0, 50, 100))  # Blue background
        
        # Title
        title = self.font.render("Hand Gesture Car Control", True, (255, 255, 255))
        title_rect = title.get_rect(center=(400, 100))
        self.screen.blit(title, title_rect)
        
        # Subtitle
        subtitle = pygame.font.Font(None, 28).render("Training Mode", True, (200, 200, 200))
        subtitle_rect = subtitle.get_rect(center=(400, 140))
        self.screen.blit(subtitle, subtitle_rect)
        
        # Status
        status = pygame.font.Font(None, 24).render("System ready - Press ESC to exit", True, (0, 255, 0))
        status_rect = status.get_rect(center=(400, 200))
        self.screen.blit(status, status_rect)
        
        # Simple animation
        import math
        time_ms = pygame.time.get_ticks()
        y_offset = int(math.sin(time_ms / 1000) * 10)
        
        # Moving rectangle
        pygame.draw.rect(self.screen, (255, 255, 0), 
                        (350, 300 + y_offset, 100, 50))
        
        # Car representation
        car_text = pygame.font.Font(None, 20).render("🚗", True, (255, 255, 255))
        car_rect = car_text.get_rect(center=(400, 325 + y_offset))
        self.screen.blit(car_text, car_rect)

    def run(self):
        """Main game loop with proper error handling"""
        setup_success = self.setup()
        
        if not setup_success:
            print("❌ Setup failed, running in error mode")
        
        # Main loop
        while self.running:
            try:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                
                # Draw appropriate screen
                if setup_success and self.screen:
                    self.draw_menu()
                elif self.screen:
                    self.draw_error_screen()
                else:
                    # If no screen available, just exit
                    print("❌ No display available, exiting...")
                    break
                
                # Update display
                if self.screen:
                    pygame.display.flip()
                
                # Control frame rate
                if self.clock:
                    self.clock.tick(60)
                else:
                    time.sleep(0.016)  # ~60 FPS fallback
                
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                traceback.print_exc()
                break
        
        self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up...")
        
        try:
            if pygame.get_init():
                pygame.quit()
                print("✅ pygame cleaned up")
        except Exception as e:
            print(f"⚠️ Error during pygame cleanup: {e}")
        
        print("👋 Goodbye!")

def test_basic_functionality():
    """Test basic functionality before starting the full game"""
    print("🧪 Running basic functionality test...")
    
    try:
        # Test pygame initialization
        print("  Testing pygame...")
        pygame.init()
        test_screen = pygame.display.set_mode((100, 100))
        pygame.quit()
        print("  ✅ pygame test passed")
        
        # Test camera access
        print("  Testing camera...")
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("  ✅ Camera test passed")
            else:
                print("  ⚠️ Camera opened but couldn't read frame")
        else:
            print("  ⚠️ Camera test failed - no camera detected")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic test failed: {e}")
        return False

def main():
    """
    Main function that starts the application with comprehensive error handling
    """
    print("🎮 Hand Gesture Car Control System - Starting...")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        print(f"   Current version: {sys.version}")
        return
    else:
        print(f"✅ Python version: {sys.version.split()[0]}")
    
    # Run basic tests first
    if not test_basic_functionality():
        print("❌ Basic functionality test failed")
        print("💡 Try running debug_main.py for detailed diagnostics")
        return
    
    # Try to import and run the game
    try:
        print("🎯 Attempting to start training mode...")
        
        # Create and run training mode
        training = TrainingMode()
        training.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Critical error: {e}")
        traceback.print_exc()
        
        # Provide helpful debugging information
        print("\n🔧 DEBUGGING INFORMATION:")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Python executable: {sys.executable}")
        print(f"   Python path: {sys.path[:3]}...")  # Show first 3 entries
        
        print("\n💡 TROUBLESHOOTING STEPS:")
        print("   1. Run: python debug_main.py")
        print("   2. Check that all dependencies are installed")
        print("   3. Make sure your webcam is working")
        print("   4. Try restarting your computer")
        
    finally:
        print("\n👋 Program ended")

if __name__ == "__main__":
    main()