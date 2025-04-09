#!/usr/bin/env python
"""
Main Game Launcher

This script serves as the main entry point for the Car Control Game.
It provides a menu to select different game modes, including the training mode.
"""

import pygame
import sys
import os
import time
from pygame.locals import *

# Import our game modes
from config import GAME_MODES, DEFAULT_GAME_MODE
from training_mode import TrainingMode

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (100, 100, 100)
LIGHT_BLUE = (173, 216, 230)
DARK_BLUE = (0, 0, 139)

class Button:
    """A simple button class for the menu."""
    
    def __init__(self, x, y, width, height, text, color, hover_color, text_color, action=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.action = action
        self.hovered = False
        
    def draw(self, screen, font):
        """Draw the button on the screen."""
        # Determine color based on hover state
        color = self.hover_color if self.hovered else self.color
        
        # Draw button
        pygame.draw.rect(screen, color, self.rect, 0, 10)
        pygame.draw.rect(screen, BLACK, self.rect, 2, 10)
        
        # Draw text
        text_surface = font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)
        
    def check_hover(self, mouse_pos):
        """Check if the mouse is hovering over the button."""
        self.hovered = self.rect.collidepoint(mouse_pos)
        return self.hovered
        
    def handle_event(self, event):
        """Handle button click events."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.hovered and self.action:
                return self.action()
        return None


class GameLauncher:
    """Main game launcher with menu to select game modes."""
    
    def __init__(self):
        """Initialize the game launcher."""
        pygame.init()
        pygame.display.set_caption("Hand Gesture Car Control Game")
        
        # Set up the screen
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Set up fonts
        self.font_large = pygame.font.Font(None, 64)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Set up buttons for game modes
        self.buttons = []
        self.create_buttons()
        
        # Create training mode instance
        self.training_mode = None
        
        # Menu background animation
        self.bg_offset = 0
        self.menu_active = True
        
    def create_buttons(self):
        """Create buttons for each game mode plus training mode."""
        button_width = 300
        button_height = 60
        start_y = 200
        button_spacing = 80
        
        # First, add the Training Mode button
        self.buttons.append(
            Button(
                SCREEN_WIDTH // 2 - button_width // 2,
                start_y,
                button_width,
                button_height,
                "Training Mode",
                LIGHT_BLUE,
                BLUE,
                WHITE,
                self.launch_training_mode
            )
        )
        
        # Then add buttons for all game modes from config
        for i, (mode_key, mode_data) in enumerate(GAME_MODES.items()):
            self.buttons.append(
                Button(
                    SCREEN_WIDTH // 2 - button_width // 2,
                    start_y + (i + 1) * button_spacing,
                    button_width,
                    button_height,
                    mode_data['name'],
                    LIGHT_BLUE,
                    BLUE,
                    WHITE,
                    lambda mode=mode_key: self.launch_game_mode(mode)
                )
            )
            
        # Add a quit button at the bottom
        self.buttons.append(
            Button(
                SCREEN_WIDTH // 2 - button_width // 2,
                start_y + (len(GAME_MODES) + 1) * button_spacing,
                button_width,
                button_height,
                "Quit",
                RED,
                (200, 0, 0),
                WHITE,
                self.quit_game
            )
        )
        
    def launch_training_mode(self):
        """Launch the training mode."""
        print("Launching Training Mode...")
        self.menu_active = False
        
        # Create and run the training mode
        self.training_mode = TrainingMode()
        
        try:
            self.training_mode.run()
        except Exception as e:
            print(f"Error in training mode: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.training_mode:
                self.training_mode.cleanup()
                self.training_mode = None
            
            # Return to menu
            self.menu_active = True
            return True
    
    def launch_game_mode(self, mode):
        """Launch the selected game mode."""
        print(f"Launching game mode: {mode}")
        # TODO: Implement actual game mode launching
        # This would connect to your main game code with the specified mode
        
        # For now, just show a simple message
        self.show_message(f"Game Mode '{GAME_MODES[mode]['name']}' not implemented yet!")
        return True
    
    def quit_game(self):
        """Quit the game."""
        return False
    
    def show_message(self, message):
        """Show a message on screen temporarily."""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        message_text = self.font_medium.render(message, True, WHITE)
        self.screen.blit(message_text, (SCREEN_WIDTH // 2 - message_text.get_width() // 2, SCREEN_HEIGHT // 2))
        
        pygame.display.flip()
        time.sleep(2)  # Show message for 2 seconds
    
    def handle_events(self):
        """Handle events for the menu."""
        mouse_pos = pygame.mouse.get_pos()
        
        # Update button hover states
        for button in self.buttons:
            button.check_hover(mouse_pos)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
            
            # Handle button clicks
            for button in self.buttons:
                result = button.handle_event(event)
                if result is not None:
                    return result
        
        return True
    
    def draw_menu(self):
        """Draw the main menu."""
        # Fill background with a gradient
        for y in range(SCREEN_HEIGHT):
            # Create a gradient from dark blue to light blue
            color_value = y / SCREEN_HEIGHT
            color = (
                int(0 + color_value * 173),
                int(0 + color_value * 216),
                int(139 + color_value * (230 - 139))
            )
            pygame.draw.line(self.screen, color, (0, y), (SCREEN_WIDTH, y))
        
        # Draw animated background pattern (grid of dots)
        self.bg_offset = (self.bg_offset + 0.5) % 50
        for x in range(0, SCREEN_WIDTH + 50, 50):
            for y in range(0, SCREEN_HEIGHT + 50, 50):
                pygame.draw.circle(
                    self.screen,
                    WHITE,
                    (int(x + self.bg_offset), int(y + self.bg_offset)),
                    2
                )
        
        # Draw title
        title_text = self.font_large.render("Hand Gesture Car Control", True, WHITE)
        title_shadow = self.font_large.render("Hand Gesture Car Control", True, BLACK)
        
        # Draw with shadow effect
        self.screen.blit(title_shadow, (SCREEN_WIDTH // 2 - title_text.get_width() // 2 + 2, 80 + 2))
        self.screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() // 2, 80))
        
        # Draw subtitle
        subtitle_text = self.font_medium.render("Select a Game Mode", True, WHITE)
        self.screen.blit(subtitle_text, (SCREEN_WIDTH // 2 - subtitle_text.get_width() // 2, 140))
        
        # Draw all buttons
        for button in self.buttons:
            button.draw(self.screen, self.font_medium)
            
        # Draw description for the hovered button
        for button in self.buttons:
            if button.hovered:
                # Find the game mode description
                description = ""
                for mode_key, mode_data in GAME_MODES.items():
                    if mode_data['name'] == button.text:
                        description = mode_data['description']
                        break
                
                # Special case for training mode and quit
                if button.text == "Training Mode":
                    description = "Practice controlling the car with hand gestures without time pressure."
                elif button.text == "Quit":
                    description = "Exit the game."
                
                # Draw the description
                if description:
                    desc_text = self.font_small.render(description, True, WHITE)
                    pygame.draw.rect(
                        self.screen,
                        DARK_BLUE,
                        (SCREEN_WIDTH // 2 - desc_text.get_width() // 2 - 10,
                         SCREEN_HEIGHT - 100,
                         desc_text.get_width() + 20,
                         40),
                        0,
                        10
                    )
                    self.screen.blit(
                        desc_text,
                        (SCREEN_WIDTH // 2 - desc_text.get_width() // 2,
                         SCREEN_HEIGHT - 90)
                    )
                break
        
        # Draw instructions at the bottom
        instructions = self.font_small.render(
            "Use your mouse to select a mode. Press ESC to quit.",
            True,
            WHITE
        )
        self.screen.blit(
            instructions,
            (SCREEN_WIDTH // 2 - instructions.get_width() // 2, SCREEN_HEIGHT - 40)
        )
    
    def run(self):
        """Main loop for the game launcher."""
        running = True
        
        while running and self.menu_active:
            # Handle events
            running = self.handle_events()
            
            # Draw menu
            self.draw_menu()
            
            # Update display
            pygame.display.flip()
            
            # Cap frame rate
            self.clock.tick(FPS)
        
        return running
    
    def cleanup(self):
        """Clean up resources."""
        pygame.quit()


def main():
    """Main function to run the game launcher."""
    launcher = GameLauncher()
    
    try:
        launcher.run()
    except Exception as e:
        print(f"Error in game launcher: {e}")
        import traceback
        traceback.print_exc()
    finally:
        launcher.cleanup()


if __name__ == "__main__":
    main()