#!/usr/bin/env python
"""
Main Game Launcher for Hand Gesture Car Control Game

This module provides the main game launcher with menu system.
"""

import pygame
import sys
import os
import time

class GameLauncher:
    """Main game launcher with menu system"""
    
    def __init__(self):
        """Initialize the game launcher"""
        pygame.init()
        
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Hand Gesture Car Control Game")
        
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Fonts
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 36)
        
        # Colors
        self.colors = {
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'blue': (0, 100, 200),
            'light_blue': (100, 150, 255),
            'green': (0, 200, 0),
            'red': (200, 0, 0),
            'yellow': (255, 255, 0),
            'gray': (128, 128, 128)
        }
        
        # Menu state
        self.current_menu = "main"
        self.selected_option = 0
        self.menu_options = {
            "main": ["Start Game", "Settings", "Instructions", "Quit"],
            "mode_select": ["Easy Mode", "Normal Mode", "Hard Mode", "Back"],
            "settings": ["Camera Settings", "Controls", "Back"]
        }
        
        # Game settings
        self.selected_mode = "normal"
        
    def handle_events(self):
        """Handle menu events"""
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
                    self.selected_option = (self.selected_option - 1) % len(self.menu_options[self.current_menu])
                
                elif event.key == pygame.K_DOWN:
                    self.selected_option = (self.selected_option + 1) % len(self.menu_options[self.current_menu])
                
                elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                    self.handle_menu_selection()
    
    def handle_menu_selection(self):
        """Handle menu option selection"""
        current_options = self.menu_options[self.current_menu]
        selected_text = current_options[self.selected_option]
        
        if self.current_menu == "main":
            if selected_text == "Start Game":
                self.current_menu = "mode_select"
                self.selected_option = 0
            elif selected_text == "Settings":
                self.current_menu = "settings"
                self.selected_option = 0
            elif selected_text == "Instructions":
                self.show_instructions()
            elif selected_text == "Quit":
                self.running = False
        
        elif self.current_menu == "mode_select":
            if selected_text == "Easy Mode":
                self.selected_mode = "easy"
                self.start_game()
            elif selected_text == "Normal Mode":
                self.selected_mode = "normal"
                self.start_game()
            elif selected_text == "Hard Mode":
                self.selected_mode = "hard"
                self.start_game()
            elif selected_text == "Back":
                self.current_menu = "main"
                self.selected_option = 0
        
        elif self.current_menu == "settings":
            if selected_text == "Camera Settings":
                self.show_camera_settings()
            elif selected_text == "Controls":
                self.show_controls()
            elif selected_text == "Back":
                self.current_menu = "main"
                self.selected_option = 0
    
    def start_game(self):
        """Start the game with selected mode"""
        print(f"🎮 Starting game in {self.selected_mode} mode...")
        
        try:
            # Import and run the game
            from game.start_game import run_game
            
            # Hide menu screen
            pygame.display.set_mode((1, 1))
            
            # Run the game
            final_score = run_game(self.selected_mode)
            
            # Restore menu screen
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Hand Gesture Car Control Game")
            
            # Show final score
            self.show_game_results(final_score)
            
        except Exception as e:
            print(f"❌ Error starting game: {e}")
            import traceback
            traceback.print_exc()
            self.show_error_message("Game Error", str(e))
    
    def show_instructions(self):
        """Show game instructions"""
        instructions = [
            "Hand Gesture Controls:",
            "",
            "• Move hand left/right to steer",
            "• Raise hand up to accelerate",
            "• Lower hand to slow down",
            "• Make a fist to brake",
            "• Thumbs up for boost",
            "• Open palm to stop",
            "",
            "Keyboard Controls (backup):",
            "• Arrow keys or WASD to move",
            "• Space for boost",
            "• ESC to pause/quit",
            "",
            "Press any key to continue..."
        ]
        
        self.show_text_screen("Instructions", instructions)
    
    def show_controls(self):
        """Show control settings"""
        controls = [
            "Control Settings:",
            "",
            "Keyboard Controls:",
            "• Arrow Keys: Movement",
            "• WASD: Alternative movement",
            "• Space: Boost",
            "• ESC: Pause/Menu",
            "• H: Help",
            "• D: Debug info",
            "• C: Camera toggle",
            "",
            "Hand Gesture Sensitivity:",
            "• Currently set to default",
            "• Calibration available in-game",
            "",
            "Press any key to continue..."
        ]
        
        self.show_text_screen("Controls", controls)
    
    def show_camera_settings(self):
        """Show camera settings"""
        camera_info = [
            "Camera Settings:",
            "",
            "• Default camera will be used",
            "• Resolution: 640x480",
            "• Frame rate: 30 FPS",
            "",
            "Camera not detected?",
            "• Check camera connections",
            "• Close other camera applications",
            "• Use keyboard controls as backup",
            "",
            "Lighting Tips:",
            "• Use good lighting",
            "• Avoid backlighting",
            "• Keep hand visible to camera",
            "",
            "Press any key to continue..."
        ]
        
        self.show_text_screen("Camera Settings", camera_info)
    
    def show_text_screen(self, title, text_lines):
        """Show a text screen with the given content"""
        waiting = True
        while waiting and self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    waiting = False
            
            # Draw background
            self.screen.fill(self.colors['black'])
            
            # Draw title
            title_surface = self.font_large.render(title, True, self.colors['white'])
            title_rect = title_surface.get_rect(center=(self.screen_width // 2, 80))
            self.screen.blit(title_surface, title_rect)
            
            # Draw text lines
            y = 150
            for line in text_lines:
                if line.strip():  # Non-empty line
                    if line.startswith("•"):
                        color = self.colors['light_blue']
                        font = self.font_small
                    elif line.endswith(":"):
                        color = self.colors['yellow']
                        font = self.font_medium
                    else:
                        color = self.colors['white']
                        font = self.font_small
                    
                    text_surface = font.render(line, True, color)
                    text_rect = text_surface.get_rect(center=(self.screen_width // 2, y))
                    self.screen.blit(text_surface, text_rect)
                
                y += 30
            
            pygame.display.flip()
            self.clock.tick(60)
    
    def show_game_results(self, score):
        """Show game results screen"""
        results = [
            f"Final Score: {int(score)}",
            "",
            "Thank you for playing!",
            "",
            "Press any key to return to menu..."
        ]
        
        self.show_text_screen("Game Complete", results)
    
    def show_error_message(self, title, message):
        """Show error message"""
        error_lines = [
            f"Error: {message}",
            "",
            "Please check the console for details.",
            "",
            "Press any key to continue..."
        ]
        
        self.show_text_screen(title, error_lines)
    
    def draw_menu(self):
        """Draw the current menu"""
        # Clear screen
        self.screen.fill(self.colors['black'])
        
        # Draw title
        title = "Hand Gesture Car Control"
        title_surface = self.font_large.render(title, True, self.colors['white'])
        title_rect = title_surface.get_rect(center=(self.screen_width // 2, 100))
        self.screen.blit(title_surface, title_rect)
        
        # Draw subtitle
        subtitle = f"Menu: {self.current_menu.replace('_', ' ').title()}"
        subtitle_surface = self.font_small.render(subtitle, True, self.colors['gray'])
        subtitle_rect = subtitle_surface.get_rect(center=(self.screen_width // 2, 140))
        self.screen.blit(subtitle_surface, subtitle_rect)
        
        # Draw menu options
        options = self.menu_options[self.current_menu]
        start_y = 250
        
        for i, option in enumerate(options):
            # Choose color based on selection
            if i == self.selected_option:
                color = self.colors['yellow']
                font = self.font_medium
                # Draw selection background
                bg_rect = pygame.Rect(
                    self.screen_width // 2 - 150,
                    start_y + i * 60 - 20,
                    300,
                    50
                )
                pygame.draw.rect(self.screen, self.colors['blue'], bg_rect, 0, 10)
            else:
                color = self.colors['white']
                font = self.font_small
            
            # Draw option text
            option_surface = font.render(option, True, color)
            option_rect = option_surface.get_rect(center=(self.screen_width // 2, start_y + i * 60))
            self.screen.blit(option_surface, option_rect)
        
        # Draw instructions
        instruction_text = "Use UP/DOWN arrows and ENTER to navigate"
        instruction_surface = self.font_small.render(instruction_text, True, self.colors['gray'])
        instruction_rect = instruction_surface.get_rect(center=(self.screen_width // 2, self.screen_height - 50))
        self.screen.blit(instruction_surface, instruction_rect)
    
    def run(self):
        """Run the game launcher"""
        print("🎮 Game Launcher Started")
        
        while self.running:
            # Handle events
            self.handle_events()
            
            # Draw menu
            self.draw_menu()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(60)
        
        # Cleanup
        pygame.quit()
        print("👋 Game Launcher Ended")
    
    def cleanup(self):
        """Cleanup resources"""
        pygame.quit()

# Main execution
if __name__ == "__main__":
    launcher = GameLauncher()
    try:
        launcher.run()
    except Exception as e:
        print(f"❌ Error in game launcher: {e}")
        import traceback
        traceback.print_exc()
    finally:
        launcher.cleanup()