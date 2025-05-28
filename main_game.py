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

# Configuration
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60

class GameLauncher:
    """Main game launcher with menu system"""
    
    def __init__(self):
        """Initialize the game launcher"""
        # Initialize pygame
        pygame.init()
        pygame.font.init()
        
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Hand Gesture Car Control - Main Menu")
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.title_font = pygame.font.Font(None, 72)
        self.menu_font = pygame.font.Font(None, 48)
        self.info_font = pygame.font.Font(None, 24)
        
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
        self.debug_mode = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def run(self):
        """Run the main menu loop"""
        self.logger.info("Starting game launcher")
        
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        
        self.cleanup()
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.current_menu == "main":
                        self.running = False
                    else:
                        self.current_menu = "main"
                        self.selected_option = 0
                
                elif event.key == pygame.K_UP:
                    self.navigate_menu(-1)
                
                elif event.key == pygame.K_DOWN:
                    self.navigate_menu(1)
                
                elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                    self.select_option()
    
    def navigate_menu(self, direction):
        """Navigate menu options"""
        if self.current_menu == "main":
            options_count = len(self.main_menu_options)
        elif self.current_menu == "difficulty":
            options_count = len(self.difficulty_options)
        elif self.current_menu == "settings":
            options_count = len(self.settings_options)
        else:
            options_count = 1
        
        self.selected_option = (self.selected_option + direction) % options_count
    
    def select_option(self):
        """Handle option selection"""
        if self.current_menu == "main":
            self.handle_main_menu_selection()
        elif self.current_menu == "difficulty":
            self.handle_difficulty_selection()
        elif self.current_menu == "settings":
            self.handle_settings_selection()
        elif self.current_menu == "instructions":
            self.current_menu = "main"
            self.selected_option = 0
    
    def handle_main_menu_selection(self):
        """Handle main menu selection"""
        option = self.main_menu_options[self.selected_option]
        
        if option == "Start Game":
            self.start_game()
        elif option == "Select Difficulty":
            self.current_menu = "difficulty"
            self.selected_option = 0
        elif option == "Settings":
            self.current_menu = "settings"
            self.selected_option = 0
        elif option == "Instructions":
            self.current_menu = "instructions"
        elif option == "Quit":
            self.running = False
    
    def handle_difficulty_selection(self):
        """Handle difficulty menu selection"""
        option = self.difficulty_options[self.selected_option]
        
        if option == "Back":
            self.current_menu = "main"
            self.selected_option = 0
        else:
            self.selected_difficulty = option.lower()
            self.current_menu = "main"
            self.selected_option = 0
            self.logger.info(f"Difficulty set to: {self.selected_difficulty}")
    
    def handle_settings_selection(self):
        """Handle settings menu selection"""
        option = self.settings_options[self.selected_option]
        
        if option == "Toggle Camera":
            self.camera_enabled = not self.camera_enabled
            self.logger.info(f"Camera {'enabled' if self.camera_enabled else 'disabled'}")
        elif option == "Test Camera":
            self.test_camera()
        elif option == "Debug Mode":
            self.debug_mode = not self.debug_mode
            self.logger.info(f"Debug mode {'enabled' if self.debug_mode else 'disabled'}")
        elif option == "Back":
            self.current_menu = "main"
            self.selected_option = 0
    
    def start_game(self):
        """Start the actual game"""
        self.logger.info(f"Starting game with difficulty: {self.selected_difficulty}")
        
        try:
            # Import the game
            from game.start_game import run_game
            
            # Hide the menu window
            pygame.display.iconify()
            
            # Set environment variables
            if not self.camera_enabled:
                os.environ['NO_CAMERA'] = '1'
            if self.debug_mode:
                os.environ['DEBUG_MODE'] = '1'
            
            # Run the game
            final_score = run_game(mode=self.selected_difficulty)
            
            # Show result
            self.show_game_result(final_score)
            
        except Exception as e:
            self.logger.error(f"Error starting game: {e}")
            self.show_error_message("Game Error", str(e))
        
        # Restore the menu window
        pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Hand Gesture Car Control - Main Menu")
    
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
        self.screen.fill(self.colors['black'])
        
        if self.current_menu == "main":
            self.draw_main_menu()
        elif self.current_menu == "difficulty":
            self.draw_difficulty_menu()
        elif self.current_menu == "settings":
            self.draw_settings_menu()
        elif self.current_menu == "instructions":
            self.draw_instructions()
        
        pygame.display.flip()
    
    def draw_main_menu(self):
        """Draw the main menu"""
        # Title
        title_text = self.title_font.render("Hand Gesture", True, self.colors['white'])
        title_text2 = self.title_font.render("Car Control", True, self.colors['light_blue'])
        
        self.screen.blit(title_text, (WINDOW_WIDTH // 2 - title_text.get_width() // 2, 80))
        self.screen.blit(title_text2, (WINDOW_WIDTH // 2 - title_text2.get_width() // 2, 150))
        
        # Menu options
        for i, option in enumerate(self.main_menu_options):
            color = self.colors['yellow'] if i == self.selected_option else self.colors['white']
            text = self.menu_font.render(option, True, color)
            y = 280 + i * 60
            self.screen.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, y))
        
        # Status info
        status_text = f"Difficulty: {self.selected_difficulty.capitalize()} | Camera: {'On' if self.camera_enabled else 'Off'}"
        status_surface = self.info_font.render(status_text, True, self.colors['gray'])
        self.screen.blit(status_surface, (10, WINDOW_HEIGHT - 30))
    
    def draw_difficulty_menu(self):
        """Draw the difficulty selection menu"""
        title_text = self.menu_font.render("Select Difficulty", True, self.colors['white'])
        self.screen.blit(title_text, (WINDOW_WIDTH // 2 - title_text.get_width() // 2, 100))
        
        # Difficulty descriptions
        descriptions = {
            "Easy": "Longer time, fewer obstacles",
            "Normal": "Balanced gameplay",
            "Hard": "Shorter time, more obstacles"
        }
        
        for i, option in enumerate(self.difficulty_options):
            if option == "Back":
                color = self.colors['yellow'] if i == self.selected_option else self.colors['white']
                text = self.menu_font.render(option, True, color)
                y = 200 + i * 80
                self.screen.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, y))
            else:
                # Highlight current difficulty
                if option.lower() == self.selected_difficulty:
                    color = self.colors['green']
                elif i == self.selected_option:
                    color = self.colors['yellow']
                else:
                    color = self.colors['white']
                
                text = self.menu_font.render(option, True, color)
                y = 200 + i * 80
                self.screen.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, y))
                
                # Description
                if option in descriptions:
                    desc_text = self.info_font.render(descriptions[option], True, self.colors['gray'])
                    self.screen.blit(desc_text, (WINDOW_WIDTH // 2 - desc_text.get_width() // 2, y + 35))
    
    def draw_settings_menu(self):
        """Draw the settings menu"""
        title_text = self.menu_font.render("Settings", True, self.colors['white'])
        self.screen.blit(title_text, (WINDOW_WIDTH // 2 - title_text.get_width() // 2, 100))
        
        for i, option in enumerate(self.settings_options):
            color = self.colors['yellow'] if i == self.selected_option else self.colors['white']
            
            # Add status indicators
            if option == "Toggle Camera":
                status = "ON" if self.camera_enabled else "OFF"
                display_text = f"{option}: {status}"
            elif option == "Debug Mode":
                status = "ON" if self.debug_mode else "OFF"
                display_text = f"{option}: {status}"
            else:
                display_text = option
            
            text = self.menu_font.render(display_text, True, color)
            y = 200 + i * 60
            self.screen.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, y))
    
    def draw_instructions(self):
        """Draw the instructions screen"""
        title_text = self.menu_font.render("Instructions", True, self.colors['white'])
        self.screen.blit(title_text, (WINDOW_WIDTH // 2 - title_text.get_width() // 2, 50))
        
        instructions = [
            "Hand Gestures:",
            "• Open Palm - Stop/Neutral",
            "• Fist - Brake",
            "• Thumbs Up - Boost",
            "• Move Hand Left/Right - Steer",
            "• Move Hand Up/Down - Throttle",
            "",
            "Keyboard Controls (Fallback):",
            "• Arrow Keys / WASD - Movement",
            "• Space - Boost",
            "• ESC - Pause/Quit",
            "",
            "Game Objective:",
            "• Avoid obstacles",
            "• Survive as long as possible",
            "• Score points by distance traveled",
            "",
            "Press any key to return to menu"
        ]
        
        y = 120
        for line in instructions:
            color = self.colors['yellow'] if line.startswith("•") else self.colors['white']
            text = self.info_font.render(line, True, color)
            self.screen.blit(text, (50, y))
            y += 25
    
    def show_game_result(self, score):
        """Show game result"""
        self.logger.info(f"Game ended with score: {score}")
        
        # Simple result display
        result_text = f"Game Over! Final Score: {int(score)}"
        self.show_info_message("Game Result", result_text)
    
    def show_error_message(self, title, message):
        """Show error message"""
        self.logger.error(f"{title}: {message}")
        # For now, just print to console
        print(f"ERROR - {title}: {message}")
    
    def show_info_message(self, title, message):
        """Show info message"""
        self.logger.info(f"{title}: {message}")
        # For now, just print to console
        print(f"INFO - {title}: {message}")
    
    def cleanup(self):
        """Clean up resources"""
        self.logger.info("Cleaning up game launcher")
        pygame.quit()

class GameLauncher:
    """Game launcher class - duplicate removed"""
    pass

def main():
    """Main function"""
    try:
        launcher = GameLauncher()
        launcher.run()
    except Exception as e:
        print(f"Error in game launcher: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()