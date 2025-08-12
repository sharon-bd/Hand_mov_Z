#!/usr/bin/env python
"""
Main Game Launcher for Hand Gesture Car Control Game

This module provides the main game launcher with menu system and configuration.
"""

import pygame
import sys
import os
import time
import logging

# Add the project root to Python path to enable imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Configuration
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60

class DebugMode:
    """Integrated debug class for system monitoring and debugging"""
    def __init__(self):
        self.debug_info = {}
        self.font = None
        self.small_font = None
        self.enabled = False
        self.performance_data = {}
        self.frame_times = []
        self.max_frame_history = 60
        
    def init_font(self):
        """Initialize font for debug display"""
        if pygame.get_init():
            try:
                self.font = pygame.font.Font(None, 24)
                self.small_font = pygame.font.Font(None, 18)
            except pygame.error:
                print("Warning: Could not initialize debug font")
    
    def update_debug_info(self, **kwargs):
        """Update debug information with current state"""
        self.debug_info.update(kwargs)
        self.debug_info['timestamp'] = time.strftime("%H:%M:%S")
        
        if self.enabled:
            debug_text = " | ".join([f"{k}: {v}" for k, v in kwargs.items()])
            print(f"[DEBUG {self.debug_info['timestamp']}] {debug_text}")
    
    def update_performance(self, frame_time):
        """Update performance metrics with frame timing data"""
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.max_frame_history:
            self.frame_times.pop(0)
        
        if self.frame_times:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            self.performance_data['fps'] = f"{fps:.1f}"
            self.performance_data['frame_time'] = f"{frame_time*1000:.1f}ms"
    
    def draw_debug_info(self, screen):
        """Draw debug information overlay on screen"""
        if not self.enabled or not self.font:
            return
        
        debug_width = 350
        debug_height = (len(self.debug_info) + len(self.performance_data) + 2) * 25 + 20
        debug_surface = pygame.Surface((debug_width, debug_height))
        debug_surface.set_alpha(180)
        debug_surface.fill((0, 0, 0))
        screen.blit(debug_surface, (10, 10))
        
        title_text = self.font.render("DEBUG INFO", True, (255, 255, 0))
        screen.blit(title_text, (15, 15))
        
        y_offset = 40
        
        for key, value in self.debug_info.items():
            text = f"{key}: {value}"
            text_surface = self.font.render(text, True, (255, 255, 255))
            screen.blit(text_surface, (15, y_offset))
            y_offset += 25
        
        pygame.draw.line(screen, (100, 100, 100), (15, y_offset), (debug_width - 5, y_offset))
        y_offset += 10
        
        perf_title = self.font.render("PERFORMANCE", True, (0, 255, 0))
        screen.blit(perf_title, (15, y_offset))
        y_offset += 25
        
        for key, value in self.performance_data.items():
            text = f"{key}: {value}"
            text_surface = self.small_font.render(text, True, (200, 255, 200))
            screen.blit(text_surface, (15, y_offset))
            y_offset += 20
    
    def toggle(self):
        """Toggle debug mode on/off"""
        self.enabled = not self.enabled
        print(f"Debug mode: {'ON' if self.enabled else 'OFF'}")
    
    def log_error(self, error_msg, exception=None):
        """Log errors to console and file"""
        timestamp = time.strftime("%H:%M:%S")
        error_text = f"[ERROR {timestamp}] {error_msg}"
        
        if exception:
            error_text += f" - {str(exception)}"
        
        print(error_text)
        
        try:
            with open("debug.log", "a", encoding='utf-8') as f:
                f.write(error_text + "\n")
        except:
            pass

class GameLauncher:
    """Main game launcher with menu system"""
    
    def __init__(self):
        """Initialize the game launcher"""
        print("DEBUG: Initializing GameLauncher...")
        
        # Initialize pygame
        pygame.init()
        pygame.font.init()
        pygame.display.init()
        
        print(f"DEBUG: Pygame initialized: {pygame.get_init()}")
        print(f"DEBUG: Font initialized: {pygame.font.get_init()}")
        print(f"DEBUG: Display initialized: {pygame.display.get_init()}")
        
        try:
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption("Hand Gesture Car Control - Main Menu")
            self.clock = pygame.time.Clock()
            print("DEBUG: Screen and clock initialized successfully")
        except Exception as e:
            print(f"ERROR: Failed to initialize display: {e}")
            raise
        
        # Initialize debug mode
        self.debug_mode = DebugMode()
        self.debug_mode.init_font()
        self.debug_mode.update_debug_info(launcher_initialized=True)
        
        # Fonts
        try:
            self.title_font = pygame.font.Font(None, 72)
            self.menu_font = pygame.font.Font(None, 48)
            self.info_font = pygame.font.Font(None, 24)
            print("DEBUG: Fonts loaded successfully")
        except Exception as e:
            print(f"ERROR: Failed to load fonts: {e}")
            raise
        
        # Colors
        self.colors = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'blue': (0, 100, 200),
            'light_blue': (100, 150, 255),
            'gray': (128, 128, 128),
            'green': (0, 200, 0),
            'red': (200, 0, 0),
            'yellow': (255, 255, 0)
        }
        
        # Menu state
        self.current_menu = "main"
        self.selected_option = 0
        self.running = True
        self._cleaned_up = False  # Track cleanup state
        
        # Debug tracking variables
        self._last_debug_menu = None
        self._last_debug_option = None
        self._difficulty_menu_drawn = False
        
        # Menu options
        self.main_menu_options = [
            "Start Game",
            "Select Difficulty",
            "Settings",
            "Instructions",
            "Quit"
        ]
        
        self.difficulty_options = [
            "Easy",
            "Normal", 
            "Hard",
            "Back"
        ]
        
        self.settings_options = [
            "Toggle Camera",
            "Test Camera",
            "Debug Mode",
            "Back"
        ]
        
        # Game settings
        self.selected_difficulty = "normal"
        self.camera_enabled = True
        self.debug_mode_enabled = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        print("DEBUG: GameLauncher initialization completed")
    
    def run(self):
        """Run the main menu loop"""
        self.logger.info("Starting game launcher")
        self.debug_mode.update_debug_info(launcher_running=True)
        
        try:
            while self.running:
                frame_start = time.time()
                
                # Check if pygame is still active before handling events
                if not pygame.get_init() or not pygame.display.get_init():
                    print("DEBUG: Pygame no longer active - exiting loop")
                    self.running = False
                    break
                
                self.handle_events()
                
                # Only update and draw if still running
                if self.running:
                    self.update()
                    self.draw()
                
                    # Draw debug information on screen
                    if self.debug_mode_enabled and self.running:
                        try:
                            self.debug_mode.draw_debug_info(self.screen)
                        except pygame.error:
                            print("DEBUG: Cannot draw debug info - display quit")
                            self.running = False
                            break
                
                if self.running:
                    try:
                        self.clock.tick(FPS)
                    except pygame.error:
                        print("DEBUG: Clock tick failed - pygame quit")
                        self.running = False
                        break
                
                # Update FPS counter
                self._update_fps()
                
                # Debug output for menu state changes
                if hasattr(self, '_last_debug_menu') and hasattr(self, '_last_debug_option'):
                    if (self._last_debug_menu != self.current_menu or 
                        self._last_debug_option != self.selected_option):
                        if self.current_menu != "main":
                            print(f"DEBUG: Menu changed to: {self.current_menu}, Selected option: {self.selected_option}")
                else:
                    # First time initialization
                    if self.current_menu != "main":
                        print(f"DEBUG: Initial menu state: {self.current_menu}, Selected option: {self.selected_option}")
                
                # Update frame performance data
                frame_time = time.time() - frame_start
                self.debug_mode.update_performance(frame_time)
        
        except KeyboardInterrupt:
            print("DEBUG: Keyboard interrupt received - exiting gracefully")
            self.running = False
        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"DEBUG: Exiting main loop - running status: {self.running}")
            self.cleanup()
    
    def handle_events(self):
        """Handle pygame events"""
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("DEBUG: QUIT event received")
                    self.debug_mode.update_debug_info(quit_requested=True)
                    self.running = False
                
                elif event.type == pygame.KEYDOWN:
                    print(f"DEBUG: Key pressed: {pygame.key.name(event.key)}")
                    
                    # F1 key to toggle debug mode
                    if event.key == pygame.K_F1:
                        self.debug_mode_enabled = not self.debug_mode_enabled
                        self.debug_mode.enabled = self.debug_mode_enabled
                        self.debug_mode.toggle()
                    
                    elif event.key == pygame.K_ESCAPE:
                        print(f"DEBUG: ESC pressed in menu: {self.current_menu}")
                        if self.current_menu == "main":
                            print("DEBUG: ESC in main menu - quitting")
                            self.debug_mode.update_debug_info(escape_quit=True)
                            self.running = False
                        else:
                            print("DEBUG: ESC in submenu - going back to main")
                            self.debug_mode.update_debug_info(menu_back=True)
                            self.current_menu = "main"
                            self.selected_option = 0
                    
                    elif event.key == pygame.K_UP:
                        # Don't navigate in instructions screen
                        if self.current_menu != "instructions":
                            self.navigate_menu(-1)
                            self.debug_mode.update_debug_info(menu_nav="up")
                    
                    elif event.key == pygame.K_DOWN:
                        # Don't navigate in instructions screen
                        if self.current_menu != "instructions":
                            self.navigate_menu(1)
                            self.debug_mode.update_debug_info(menu_nav="down")
                    
                    elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                        if self.current_menu == "instructions":
                            # Any key in instructions returns to main menu
                            self.current_menu = "main"
                            self.selected_option = 0
                        else:
                            print(f"DEBUG: Enter/Space pressed - selecting option {self.selected_option}")
                            self.debug_mode.update_debug_info(menu_select=self.selected_option)
                            self.select_option()
                    
                    else:
                        # Any other key in instructions screen returns to main menu
                        if self.current_menu == "instructions":
                            print("DEBUG: Key pressed in instructions - returning to main menu")
                            self.current_menu = "main"
                            self.selected_option = 0
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        mouse_x, mouse_y = event.pos
                        clicked_option = self.get_menu_option_at_pos(mouse_x, mouse_y)
                        if clicked_option != -1:
                            print(f"DEBUG: Mouse clicked on option {clicked_option}")
                            self.selected_option = clicked_option
                            self.debug_mode.update_debug_info(menu_select=self.selected_option, mouse_click=True)
                            self.select_option()
                
                elif event.type == pygame.MOUSEMOTION:
                    mouse_x, mouse_y = event.pos
                    hovered_option = self.get_menu_option_at_pos(mouse_x, mouse_y)
                    if hovered_option != -1 and hovered_option != self.selected_option:
                        self.selected_option = hovered_option
                        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
                    elif hovered_option == -1:
                        pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
        except pygame.error as e:
            if "display Surface quit" in str(e) or "video system not initialized" in str(e):
                print("DEBUG: Pygame display quit during event handling")
                self.running = False
            else:
                print(f"ERROR in handle_events - pygame error: {e}")
                self.running = False
        except Exception as e:
            print(f"ERROR in handle_events: {e}")
            import traceback
            traceback.print_exc()
            self.running = False
    
    def get_menu_option_at_pos(self, mouse_x, mouse_y):
        """Get menu option at mouse position, returns -1 if none"""
        if self.current_menu == "main":
            options = self.main_menu_options
            start_y = 280
            item_spacing = 60
        elif self.current_menu == "difficulty":
            options = self.difficulty_options
            start_y = 200
            item_spacing = 80  # Different spacing for difficulty menu
        elif self.current_menu == "settings":
            options = self.settings_options
            start_y = 200
            item_spacing = 60
        else:
            return -1
        
        # Check if mouse is over any menu option
        for i, option in enumerate(options):
            option_y = start_y + i * item_spacing
            option_height = 50  # Approximate height for click area
            
            if option_y <= mouse_y <= option_y + option_height:
                # Check if mouse is roughly in the center horizontal area
                if WINDOW_WIDTH // 4 <= mouse_x <= 3 * WINDOW_WIDTH // 4:
                    return i
        
        return -1

    def navigate_menu(self, direction):
        """Navigate menu options"""
        # Only print debug if this is actually a change caused by keyboard
        should_debug = True
        
        if self.current_menu == "main":
            options_count = len(self.main_menu_options)
        elif self.current_menu == "difficulty":
            options_count = len(self.difficulty_options)
        elif self.current_menu == "settings":
            options_count = len(self.settings_options)
        else:
            options_count = 1
        
        old_selection = self.selected_option
        self.selected_option = (self.selected_option + direction) % options_count
        
        if should_debug and old_selection != self.selected_option:
            print(f"DEBUG: navigate_menu - direction: {direction}, menu: {self.current_menu}")
            print(f"DEBUG: Selection changed from {old_selection} to {self.selected_option} (max: {options_count-1})")
    
    def select_option(self):
        """Handle option selection"""
        print(f"DEBUG: select_option called - Menu: {self.current_menu}, Option: {self.selected_option}")
        
        try:
            if self.current_menu == "main":
                self.handle_main_menu_selection()
            elif self.current_menu == "difficulty":
                self.handle_difficulty_selection()
            elif self.current_menu == "settings":
                self.handle_settings_selection()
            elif self.current_menu == "instructions":
                self.current_menu = "main"
                self.selected_option = 0
        except Exception as e:
            print(f"ERROR in select_option: {e}")
            import traceback
            traceback.print_exc()
    
    def handle_main_menu_selection(self):
        """Handle main menu selection"""
        option = self.main_menu_options[self.selected_option]
        print(f"DEBUG: handle_main_menu_selection - Option: {option}, Index: {self.selected_option}")
        
        if option == "Start Game":
            self.start_game()
        elif option == "Select Difficulty":
            print("DEBUG: Switching to difficulty menu")
            self.current_menu = "difficulty"
            self.selected_option = 0
            # Reset the debug flag for drawing
            self._difficulty_menu_drawn = False
            print(f"DEBUG: After switch - Menu: {self.current_menu}, Selected: {self.selected_option}")
        elif option == "Settings":
            self.current_menu = "settings"
            self.selected_option = 0
        elif option == "Instructions":
            self.current_menu = "instructions"
        elif option == "Quit":
            print("DEBUG: Quit selected")
            self.running = False
    
    def handle_difficulty_selection(self):
        """Handle difficulty menu selection"""
        option = self.difficulty_options[self.selected_option]
        
        if option == "Back":
            self.current_menu = "main"
            self.selected_option = 0
            # Reset debug flag when leaving difficulty menu
            self._difficulty_menu_drawn = False
        else:
            # Set the difficulty, stay in difficulty menu to show the change
            self.selected_difficulty = option.lower()
            self.logger.info(f"Difficulty set to: {self.selected_difficulty}")
            print(f"DEBUG: Difficulty changed to: {self.selected_difficulty}")
            # Don't change menu, just show the updated difficulty setting
    
    def handle_settings_selection(self):
        """Handle settings menu selection"""
        option = self.settings_options[self.selected_option]
        
        if option == "Toggle Camera":
            self.camera_enabled = not self.camera_enabled
            self.debug_mode.update_debug_info(camera_enabled=self.camera_enabled)
            self.logger.info(f"Camera {'enabled' if self.camera_enabled else 'disabled'}")
        elif option == "Test Camera":
            self.test_camera()
        elif option == "Debug Mode":
            self.debug_mode_enabled = not self.debug_mode_enabled
            self.debug_mode.enabled = self.debug_mode_enabled 
            self.debug_mode.update_debug_info(debug_mode_toggled=self.debug_mode_enabled)
            self.logger.info(f"Debug mode {'enabled' if self.debug_mode_enabled else 'disabled'}")
        elif option == "Back":
            self.current_menu = "main"
            self.selected_option = 0
    
    def start_game(self):
        """Start the actual game - WITH DEBUG INTEGRATION"""
        self.debug_mode.update_debug_info(action="starting_game", difficulty=self.selected_difficulty)
        self.logger.info(f"Starting game with difficulty: {self.selected_difficulty}")
        
        try:
            # Initialize debug mode for game
            self.debug_mode.enabled = self.debug_mode_enabled
            self.debug_mode.init_font()
            self.debug_mode.update_debug_info(
                game_starting=True,
                difficulty=self.selected_difficulty,
                camera_enabled=self.camera_enabled
            )
            
            # Create __init__.py if missing
            game_init_file = os.path.join(current_dir, 'game', '__init__.py')
            if not os.path.exists(game_init_file):
                self.debug_mode.update_debug_info(creating_init_file=True)
                print("⚠️ Creating missing __init__.py file...")
                os.makedirs(os.path.dirname(game_init_file), exist_ok=True)
                with open(game_init_file, 'w') as f:
                    f.write('# Game package init file\n')
            
            # Import the game
            self.debug_mode.update_debug_info(importing_game_module=True)
            print("🔄 Importing game module...")
            from game.start_game import run_game
            print("✅ Game module imported successfully")
            
            # Hide the menu window
            pygame.display.iconify()
            
            # Set environment variables
            if not self.camera_enabled:
                os.environ['NO_CAMERA'] = '1'
            if self.debug_mode_enabled:
                os.environ['DEBUG_MODE'] = '1'
            
            self.debug_mode.update_debug_info(
                environment_set=True,
                no_camera=not self.camera_enabled,
                debug_env=self.debug_mode_enabled
            )
            print(f"🎮 Starting game with mode: {self.selected_difficulty}")
            
            # Run the game
            final_score = run_game(mode=self.selected_difficulty)
            
            self.debug_mode.update_debug_info(game_finished=True, final_score=final_score)
            print(f"🏆 Game finished with score: {final_score}")
            
            # Show result
            self.show_game_result(final_score)
            
        except Exception as e:
            self.debug_mode.log_error("Error starting game", e)
            self.logger.error(f"Error starting game: {e}")
            print(f"❌ Full error details: {e}")
            import traceback
            traceback.print_exc()
            self.show_error_message("Game Error", str(e))
        
        finally:
            # Always try to restore the menu window
            try:
                pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
                pygame.display.set_caption("Hand Gesture Car Control - Main Menu")
                self.debug_mode.update_debug_info(menu_restored=True)
                print("🔄 Menu window restored")
            except Exception as e:
                self.debug_mode.log_error("Error restoring menu window", e)
                self.logger.error(f"Error restoring menu window: {e}")
    
    def test_camera(self):
        """Test camera functionality"""
        self.logger.info("Testing camera...")
        
        try:
            import cv2
            
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise Exception("Could not open camera")
            
            # Show camera test for 3 seconds
            start_time = time.time()
            while time.time() - start_time < 3.0:
                ret, frame = cap.read()
                if ret:
                    cv2.imshow("Camera Test", frame)
                    cv2.waitKey(1)
                
                # Handle pygame events to prevent freezing
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        break
            
            cap.release()
            cv2.destroyAllWindows()
            
            self.show_info_message("Camera Test", "Camera test completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Camera test failed: {e}")
            self.show_error_message("Camera Test Failed", str(e))
    
    def update(self):
        """Update game state"""
        pass
    
    def draw(self):
        """Draw the current screen"""
        try:
            # Check if pygame display is still valid
            if not pygame.get_init() or not pygame.display.get_init() or not self.screen:
                print("DEBUG: Pygame display is not initialized - skipping draw")
                self.running = False
                return
            
            # Check if the display surface is still valid
            try:
                self.screen.get_size()  # Test if surface is valid
            except pygame.error:
                print("DEBUG: Display surface is invalid - stopping")
                self.running = False
                return
            
            self.screen.fill(self.colors['black'])
            
            if self.current_menu == "main":
                self.draw_main_menu()
            elif self.current_menu == "difficulty":
                # Print debug info only once when entering difficulty menu
                if not hasattr(self, '_difficulty_menu_drawn') or not self._difficulty_menu_drawn:
                    print(f"DEBUG: Drawing difficulty menu with {len(self.difficulty_options)} options")
                    self._difficulty_menu_drawn = True
                self.draw_difficulty_menu()
            elif self.current_menu == "settings":
                self.draw_settings_menu()
            elif self.current_menu == "instructions":
                self.draw_instructions()
            else:
                # Reset flag when leaving difficulty menu
                if hasattr(self, '_difficulty_menu_drawn'):
                    self._difficulty_menu_drawn = False
            
            pygame.display.flip()
        except pygame.error as e:
            if "display Surface quit" in str(e):
                print("DEBUG: Display surface quit - stopping gracefully")
                self.running = False
            else:
                print(f"ERROR in draw() - pygame error: {e}")
                self.running = False
        except Exception as e:
            print(f"ERROR in draw(): {e}")
            import traceback
            traceback.print_exc()
            self.running = False
    
    def draw_main_menu(self):
        """Draw the main menu"""
        # Title
        title_text = self.title_font.render("Hand Gesture", True, self.colors['white'])
        title_text2 = self.title_font.render("Car Control", True, self.colors['light_blue'])
        
        self.screen.blit(title_text, (WINDOW_WIDTH // 2 - title_text.get_width() // 2, 80))
        self.screen.blit(title_text2, (WINDOW_WIDTH // 2 - title_text2.get_width() // 2, 150))
        
        # Menu options with hover effect
        for i, option in enumerate(self.main_menu_options):
            if i == self.selected_option:
                # Calculate text width for proper highlight sizing
                text_width = self.menu_font.size(option)[0]
                # Draw background highlight for selected option with proper width
                highlight_rect = pygame.Rect(WINDOW_WIDTH // 2 - text_width // 2 - 20, 280 + i * 60 - 5, text_width + 40, 50)
                pygame.draw.rect(self.screen, (50, 50, 100), highlight_rect)
                pygame.draw.rect(self.screen, self.colors['yellow'], highlight_rect, 2)
                color = self.colors['yellow']
            else:
                color = self.colors['white']
            
            text = self.menu_font.render(option, True, color)
            y = 280 + i * 60
            self.screen.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, y))
        
        # Status info - positioned higher to avoid overlap
        status_text = f"Difficulty: {self.selected_difficulty.capitalize()} | Camera: {'On' if self.camera_enabled else 'Off'}"
        status_surface = self.info_font.render(status_text, True, self.colors['gray'])
        self.screen.blit(status_surface, (10, WINDOW_HEIGHT - 60))  # Moved up
        
        # Add file path info at bottom - positioned higher
        game_file_path = os.path.join(current_dir, 'game', 'start_game.py')
        file_exists = os.path.exists(game_file_path)
        file_status = f"Game file: {'Found' if file_exists else 'Missing'} at game/start_game.py"
        file_surface = self.info_font.render(file_status, True, self.colors['green'] if file_exists else self.colors['red'])
        self.screen.blit(file_surface, (10, WINDOW_HEIGHT - 90))  # Moved up
        
        # Add control instructions - split into two lines for better readability
        control_line1 = "Use ↑↓ keys or mouse to navigate"
        control_line2 = "Enter/Space or click to select"
        
        control_surface1 = self.info_font.render(control_line1, True, self.colors['gray'])
        control_surface2 = self.info_font.render(control_line2, True, self.colors['gray'])
        
        # Center both lines
        self.screen.blit(control_surface1, (WINDOW_WIDTH // 2 - control_surface1.get_width() // 2, WINDOW_HEIGHT - 40))
        self.screen.blit(control_surface2, (WINDOW_WIDTH // 2 - control_surface2.get_width() // 2, WINDOW_HEIGHT - 20))
    
    def draw_difficulty_menu(self):
        """Draw the difficulty selection menu"""
        try:
            title_text = self.menu_font.render("Select Difficulty", True, self.colors['white'])
            self.screen.blit(title_text, (WINDOW_WIDTH // 2 - title_text.get_width() // 2, 100))
            
            # Difficulty descriptions
            descriptions = {
                "Easy": "Longer time, fewer obstacles",
                "Normal": "Balanced gameplay",
                "Hard": "Shorter time, more obstacles"
            }
            
            for i, option in enumerate(self.difficulty_options):
                try:
                    if option == "Back":
                        if i == self.selected_option:
                            # Calculate text width for proper highlight sizing
                            text_width = self.menu_font.size(option)[0]
                            # Draw background highlight with proper width
                            highlight_rect = pygame.Rect(WINDOW_WIDTH // 2 - text_width // 2 - 20, 200 + i * 80 - 5, text_width + 40, 50)
                            pygame.draw.rect(self.screen, (50, 50, 100), highlight_rect)
                            pygame.draw.rect(self.screen, self.colors['yellow'], highlight_rect, 2)
                            color = self.colors['yellow']
                        else:
                            color = self.colors['white']
                        text = self.menu_font.render(option, True, color)
                        y = 200 + i * 80
                        self.screen.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, y))
                    else:
                        # Highlight current difficulty
                        if option.lower() == self.selected_difficulty:
                            # Current difficulty - show in green with special marking
                            color = self.colors['green']
                            display_text = f"★ {option} ★ (Current)"
                        elif i == self.selected_option:
                            # Calculate text width for proper highlight sizing
                            display_text = option
                            text_width = self.menu_font.size(display_text)[0]
                            # Draw background highlight for selected option with proper width
                            highlight_rect = pygame.Rect(WINDOW_WIDTH // 2 - text_width // 2 - 20, 200 + i * 80 - 5, text_width + 40, 50)
                            pygame.draw.rect(self.screen, (50, 50, 100), highlight_rect)
                            pygame.draw.rect(self.screen, self.colors['yellow'], highlight_rect, 2)
                            color = self.colors['yellow']
                        else:
                            color = self.colors['white']
                            display_text = option
                        
                        text = self.menu_font.render(display_text, True, color)
                        y = 200 + i * 80
                        self.screen.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, y))
                        
                        # Description
                        if option in descriptions:
                            desc_text = self.info_font.render(descriptions[option], True, self.colors['gray'])
                            self.screen.blit(desc_text, (WINDOW_WIDTH // 2 - desc_text.get_width() // 2, y + 35))
                except Exception as e:
                    print(f"Error drawing difficulty option {i}: {e}")
                    # Continue with next option
                    continue
            
            # Add helpful instructions
            instruction_text = "Select difficulty level • Current choice marked with ★"
            instruction_surface = self.info_font.render(instruction_text, True, self.colors['light_blue'])
            self.screen.blit(instruction_surface, (WINDOW_WIDTH // 2 - instruction_surface.get_width() // 2, WINDOW_HEIGHT - 60))
            
            # Add control instructions
            control_text = "Use ↑↓ keys or mouse to navigate • Enter/Space or click to select • ESC to go back"
            control_surface = self.info_font.render(control_text, True, self.colors['gray'])
            self.screen.blit(control_surface, (WINDOW_WIDTH // 2 - control_surface.get_width() // 2, WINDOW_HEIGHT - 30))
            
        except Exception as e:
            print(f"ERROR in draw_difficulty_menu: {e}")
            import traceback
            traceback.print_exc()
            # Return to main menu if there's an error
            self.current_menu = "main"
            self.selected_option = 0
    
    def draw_settings_menu(self):
        """Draw the settings menu - WITH DEBUG STATUS"""
        title_text = self.menu_font.render("Settings", True, self.colors['white'])
        self.screen.blit(title_text, (WINDOW_WIDTH // 2 - title_text.get_width() // 2, 100))
        
        for i, option in enumerate(self.settings_options):
            # Add status indicators
            if option == "Toggle Camera":
                status = "ON" if self.camera_enabled else "OFF"
                display_text = f"{option}: {status}"
            elif option == "Debug Mode":
                status = "ON" if self.debug_mode_enabled else "OFF"
                display_text = f"{option}: {status}"
            else:
                display_text = option
            
            if i == self.selected_option:
                # Calculate text width for proper highlight sizing
                text_width = self.menu_font.size(display_text)[0]
                # Draw background highlight with proper width
                highlight_rect = pygame.Rect(WINDOW_WIDTH // 2 - text_width // 2 - 20, 200 + i * 60 - 5, text_width + 40, 50)
                pygame.draw.rect(self.screen, (50, 50, 100), highlight_rect)
                pygame.draw.rect(self.screen, self.colors['yellow'], highlight_rect, 2)
                color = self.colors['yellow']
            else:
                color = self.colors['white']
            
            text = self.menu_font.render(display_text, True, color)
            y = 200 + i * 60
            self.screen.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, y))
        
        # Add F1 key instruction if debug mode is enabled
        if self.debug_mode_enabled:
            debug_info = self.info_font.render("Press F1 to toggle debug mode", True, self.colors['gray'])
            self.screen.blit(debug_info, (10, WINDOW_HEIGHT - 70))
        
        # Add control instructions
        control_text = "Use ↑↓ keys or mouse to navigate • Enter/Space or click to select • ESC to go back"
        control_surface = self.info_font.render(control_text, True, self.colors['gray'])
        self.screen.blit(control_surface, (WINDOW_WIDTH // 2 - control_surface.get_width() // 2, WINDOW_HEIGHT - 30))

    def draw_instructions(self):
        """Draw the instructions screen with debug features information"""
        title_text = self.menu_font.render("Instructions", True, self.colors['white'])
        self.screen.blit(title_text, (WINDOW_WIDTH // 2 - title_text.get_width() // 2, 20))
        
        # Split instructions into two columns
        left_column = [
            "Hand Gestures:",
            "• Open Palm - Stop/Neutral",
            "• Fist - Brake", 
            "• Thumbs Up - Boost",
            "• Move Hand Left/Right - Steer",
            "• Move Hand Up/Down - Throttle",
            "",
            "Keyboard Controls:",
            "• Arrow Keys / WASD - Movement",
            "• Space - Boost",
            "• ESC - Pause/Quit",
            "• F1 - Toggle Debug Mode"
        ]
        
        right_column = [
            "Debug Features:",
            "• Real-time performance monitoring",
            "• Game state visualization", 
            "• Error logging and tracking",
            "",
            "Game Objective:",
            "• Avoid obstacles",
            "• Survive as long as possible", 
            "• Score points by distance traveled",
            "",
            "Tips:",
            "• Use hand gestures for natural control",
            "• Press F1 to see debug information"
        ]
        
        # Draw left column
        y = 60
        line_height = 20  # Smaller line height
        left_x = 50
        
        for line in left_column:
            if line.startswith("Hand Gestures:") or line.startswith("Keyboard Controls:"):
                color = self.colors['light_blue']
                font = self.info_font  # Use smaller font for headers
            elif line.startswith("• F1"):
                color = self.colors['yellow']
                font = self.info_font
            elif line.startswith("•"):
                color = self.colors['white']
                font = self.info_font
            else:
                color = self.colors['gray']
                font = self.info_font
            
            if line.strip():  # Don't render empty lines
                text = font.render(line, True, color)
                self.screen.blit(text, (left_x, y))
            y += line_height
        
        # Draw right column
        y = 60
        right_x = WINDOW_WIDTH // 2 + 20  # Move closer to center
        
        for line in right_column:
            if line.startswith("Debug Features:") or line.startswith("Game Objective:") or line.startswith("Tips:"):
                color = self.colors['green']
                font = self.info_font  # Use smaller font for headers
            elif line.startswith("•"):
                color = self.colors['white']
                font = self.info_font
            else:
                color = self.colors['gray']
                font = self.info_font
            
            if line.strip():  # Don't render empty lines
                text = font.render(line, True, color)
                self.screen.blit(text, (right_x, y))
            y += line_height
        
        # Add "Press any key" instruction at the bottom
        return_text = "Press any key to return to menu"
        return_surface = self.info_font.render(return_text, True, self.colors['yellow'])  # Use smaller font
        self.screen.blit(return_surface, (WINDOW_WIDTH // 2 - return_surface.get_width() // 2, WINDOW_HEIGHT - 30))
    
    def show_game_result(self, score):
        """Show game result"""
        self.logger.info(f"Game ended with score: {score}")
        
        # Handle None score safely
        display_score = int(score) if score is not None else 0
        
        # Simple message instead of complex overlay
        print(f"🎉 Game Over! Final Score: {display_score}")
        print("Press any key to continue...")
        return  # Skip the problematic pygame rendering
        
        # Ensure font is initialized
        if not pygame.font.get_init():
            pygame.font.init()
        
        # Create a simple result display overlay
        overlay = pygame.Surface((400, 200))
        overlay.fill((50, 50, 50))
        pygame.draw.rect(overlay, (255, 255, 255), (0, 0, 400, 200), 2)
        
        result_font = pygame.font.Font(None, 36)
        result_text = result_font.render(f"Game Over!", True, (255, 255, 255))
        score_text = result_font.render(f"Final Score: {display_score}", True, (255, 215, 0))
        continue_text = self.info_font.render("Press any key to continue...", True, (200, 200, 200))
        
        overlay.blit(result_text, (200 - result_text.get_width() // 2, 50))
        overlay.blit(score_text, (200 - score_text.get_width() // 2, 100))
        overlay.blit(continue_text, (200 - continue_text.get_width() // 2, 150))
        
        # Display the overlay
        self.screen.blit(overlay, (WINDOW_WIDTH // 2 - 200, WINDOW_HEIGHT // 2 - 100))
        pygame.display.flip()
        
        # Wait for key press
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN or event.type == pygame.QUIT:
                    waiting = False
    
    def show_error_message(self, title, message):
        """Show error message"""
        self.logger.error(f"{title}: {message}")
        
        # Simple console message instead of pygame overlay
        print(f"❌ {title}: {message}")
        print("Check console for details.")
        return  # Skip the problematic pygame rendering
        
        # Ensure font is initialized
        if not pygame.font.get_init():
            pygame.font.init()
        
        # Create error message overlay
        overlay = pygame.Surface((600, 350))
        overlay.fill((50, 50, 50))
        pygame.draw.rect(overlay, (255, 0, 0), (0, 0, 600, 350), 2)
        
        title_font = pygame.font.Font(None, 32)
        msg_font = pygame.font.Font(None, 20)
        
        title_text = title_font.render(title, True, (255, 0, 0))
        # Split long messages into multiple lines
        msg_lines = []
        words = message.split()
        current_line = ""
        for word in words:
            test_line = current_line + word + " "
            if len(test_line) > 70:
                if current_line:
                    msg_lines.append(current_line.strip())
                current_line = word + " "
            else:
                current_line = test_line
        if current_line:
            msg_lines.append(current_line.strip())
        
        continue_text = msg_font.render("Press any key to continue...", True, (255, 255, 0))
        
        overlay.blit(title_text, (300 - title_text.get_width() // 2, 20))
        
        y_offset = 70
        for line in msg_lines:
            line_text = msg_font.render(line, True, (200, 200, 200))
            overlay.blit(line_text, (300 - line_text.get_width() // 2, y_offset))
            y_offset += 25
        
        overlay.blit(continue_text, (300 - continue_text.get_width() // 2, 300))
        
        # Display the overlay
        self.screen.blit(overlay, (WINDOW_WIDTH // 2 - 300, WINDOW_HEIGHT // 2 - 175))
        pygame.display.flip()
        
        # Wait for key press
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN or event.type == pygame.QUIT:
                    waiting = False
    
    def show_info_message(self, title, message):
        """Show info message"""
        self.logger.info(f"{title}: {message}")
        
        # Ensure font is initialized
        if not pygame.font.get_init():
            pygame.font.init()
        
        # Create info message overlay
        overlay = pygame.Surface((400, 200))
        overlay.fill((50, 50, 50))
        pygame.draw.rect(overlay, (0, 200, 0), (0, 0, 400, 200), 2)
        
        title_font = pygame.font.Font(None, 32)
        msg_font = pygame.font.Font(None, 24)
        
        title_text = title_font.render(title, True, (0, 200, 0))
        msg_text = msg_font.render(message, True, (200, 200, 200))
        continue_text = msg_font.render("Press any key to continue...", True, (255, 255, 0))
        
        overlay.blit(title_text, (200 - title_text.get_width() // 2, 50))
        overlay.blit(msg_text, (200 - msg_text.get_width() // 2, 100))
        overlay.blit(continue_text, (200 - continue_text.get_width() // 2, 150))
        
        # Display the overlay
        self.screen.blit(overlay, (WINDOW_WIDTH // 2 - 200, WINDOW_HEIGHT // 2 - 100))
        pygame.display.flip()
        
        # Wait for key press
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN or event.type == pygame.QUIT:
                    waiting = False
    
    def cleanup(self):
        """Clean up resources"""
        print("DEBUG: cleanup() called")
        
        # Prevent multiple cleanup calls
        if hasattr(self, '_cleaned_up') and self._cleaned_up:
            print("DEBUG: cleanup() already called - skipping")
            return
        
        self.logger.info("Cleaning up game launcher")
        try:
            # Mark as cleaned up before doing anything
            self._cleaned_up = True
            
            # Stop the running flag
            self.running = False
            
            # Quit pygame if it's still active
            if pygame.get_init():
                pygame.quit()
                print("DEBUG: pygame.quit() completed")
            else:
                print("DEBUG: pygame already quit")
        except Exception as e:
            print(f"ERROR: Error during cleanup: {e}")
            self.logger.error(f"Error during cleanup: {e}")

    def main_game_loop(self):
        """Main game loop with integrated debugging features"""
        clock = pygame.time.Clock()
        running = True
        game_start_time = time.time()
        
        # Use the integrated debug mode instead
        self.debug_mode.init_font()
        
        while running:
            current_time = time.time()
            game_duration = current_time - game_start_time
            
            # Update debug information
            self.debug_mode.update_debug_info(
                game_time=f"{game_duration:.1f}s",
                car_x=f"{getattr(self, 'car', {}).get('x', 0):.1f}",
                car_y=f"{getattr(self, 'car', {}).get('y', 0):.1f}",
                car_speed=f"{getattr(self, 'car', {}).get('speed', 0):.2f}",
                obstacles=len(getattr(self, 'obstacles', [])),
                collisions=getattr(self, 'collision_count', 0)
            )
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_d:  # Toggle debug mode
                        self.debug_mode.enabled = not self.debug_mode.enabled
                    elif event.key == pygame.K_r:  # Reset game
                        self.reset_game()
            
            # Update game state
            if hasattr(self, 'update_game'):
                self.update_game()
            
            # Check game over conditions only after minimum time
            if game_duration > 5.0:  # At least 5 seconds of gameplay
                if hasattr(self, 'check_game_over_conditions') and self.check_game_over_conditions():
                    print(f"Game Over after {game_duration:.1f} seconds")
                    break
            
            # Draw game and debug info
            if hasattr(self, 'draw_game'):
                self.draw_game()
            self.debug_mode.draw_debug_info(self.screen)
            
            pygame.display.flip()
            clock.tick(60)  # 60 FPS
        
        pygame.quit()

    def reset_game(self):
        """Reset game to initial state"""
        if hasattr(self, 'car'):
            self.car.x = getattr(self, 'screen_width', 800) // 2
            self.car.y = getattr(self, 'screen_height', 600) // 2
            self.car.speed = 0.3
            self.car.direction = 0

def main():
    """Main function"""
    launcher = None
    try:
        print("🎮 Hand Gesture Car Control System - SYNCHRONIZED VERSION")
        print("=" * 60)
        print(f"Current directory: {current_dir}")
        
        # Check if pygame is working
        print(f"Pygame initialized: {pygame.get_init()}")
        print(f"Display surface exists: {pygame.display.get_surface() is not None}")
        
        # Check if game file exists
        game_file = os.path.join(current_dir, 'game', 'start_game.py')
        print(f"Game file exists: {os.path.exists(game_file)} at {game_file}")
        
        print("DEBUG: Creating GameLauncher...")
        launcher = GameLauncher()
        print("DEBUG: Starting launcher.run()...")
        launcher.run()
        print("DEBUG: launcher.run() completed")
        
    except KeyboardInterrupt:
        print("DEBUG: Keyboard interrupt in main - exiting gracefully")
    except Exception as e:
        print(f"Error in game: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            if launcher is not None and not getattr(launcher, '_cleaned_up', False):
                print("DEBUG: Calling launcher.cleanup() from main")
                launcher.cleanup()
            else:
                print("DEBUG: Launcher already cleaned up or None")
        except Exception as e:
            print(f"Error in cleanup: {e}")
        
        print("DEBUG: Main function completed")

if __name__ == "__main__":
    main()