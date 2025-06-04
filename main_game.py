#!/usr/bin/env python
"""
Main Game Launcher for Hand Gesture Car Control Game
"""

import pygame
import sys
import os

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
            'yellow': (255, 255, 0),
            'green': (0, 200, 0),
            'red': (200, 0, 0),
            'hover': (100, 100, 100)  # צבע hover לעכבר
        }
        
        # Menu state
        self.current_menu = "main"
        self.selected_option = 0
        self.menu_options = {
            "main": ["Start Game", "Instructions", "Quit"]
        }
        
        self.selected_mode = "normal"
        
        # Mouse support
        self.mouse_pos = (0, 0)
        self.menu_rects = []  # רשימה של rectangles עבור כל אפשרות בתפריט
    
    def handle_events(self):
        """Handle menu events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                
                elif event.key == pygame.K_UP:
                    self.selected_option = (self.selected_option - 1) % len(self.menu_options[self.current_menu])
                
                elif event.key == pygame.K_DOWN:
                    self.selected_option = (self.selected_option + 1) % len(self.menu_options[self.current_menu])
                
                elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                    self.handle_menu_selection()
            
            # הוספת תמיכה בעכבר
            elif event.type == pygame.MOUSEMOTION:
                self.mouse_pos = event.pos
                self.update_selected_option_by_mouse()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # לחיצה שמאלית
                    self.mouse_pos = event.pos
                    if self.update_selected_option_by_mouse():
                        self.handle_menu_selection()
    
    def update_selected_option_by_mouse(self):
        """עדכון האפשרות הנבחרת על סמך מיקום העכבר"""
        for i, rect in enumerate(self.menu_rects):
            if rect.collidepoint(self.mouse_pos):
                if self.selected_option != i:
                    self.selected_option = i
                return True
        return False
    
    def handle_menu_selection(self):
        """Handle menu option selection"""
        current_options = self.menu_options[self.current_menu]
        selected_text = current_options[self.selected_option]
        
        if selected_text == "Start Game":
            self.start_game()
        elif selected_text == "Instructions":
            self.show_instructions()
        elif selected_text == "Quit":
            self.running = False
    
    def start_game(self):
        """Start the game"""
        print("🎮 Starting game...")
        
        try:
            # First try to import the run_game function
            try:
                from game.start_game import run_game
                # Run game with the default normal mode
                run_game("normal")
                
            except ImportError as e:
                print(f"⚠️ Warning: Some game modules could not be imported: {e}")
                
                # Fallback to importing Game class directly
                try:
                    from game.start_game import Game
                    
                    # Create and run the game
                    game = Game()
                    try:
                        game.run()
                    finally:
                        # Make sure to call cleanup if it exists
                        if hasattr(game, 'cleanup'):
                            game.cleanup()
                            
                except ImportError:
                    print("❌ Critical: Could not load core game modules")
                
        except Exception as e:
            print(f"❌ Error starting game: {e}")
            import traceback
            traceback.print_exc()
    
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
            "",
            "Keyboard Controls:",
            "• Arrow Keys / WASD: Movement",
            "• Space: Boost",
            "• P: Pause",
            "• M: Mute",
            "• H: Help",
            "• ESC: Exit",
            "",
            "Press any key to continue..."
        ]
        
        self.show_text_screen("Instructions", instructions)
    
    def show_text_screen(self, title, text_lines):
        """Show a text screen"""
        waiting = True
        while waiting and self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    waiting = False
                elif event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                    waiting = False
            
            self.screen.fill(self.colors['black'])
            
            # Draw title
            title_surface = self.font_large.render(title, True, self.colors['white'])
            title_rect = title_surface.get_rect(center=(self.screen_width // 2, 80))
            self.screen.blit(title_surface, title_rect)
            
            # Draw text lines
            y = 150
            for line in text_lines:
                if line.strip():
                    color = self.colors['white']
                    font = self.font_small
                    
                    if line.startswith("•"):
                        color = self.colors['yellow']
                    elif "Controls:" in line:
                        color = self.colors['green']
                        font = self.font_medium
                    
                    text_surface = font.render(line, True, color)
                    text_rect = text_surface.get_rect(center=(self.screen_width // 2, y))
                    self.screen.blit(text_surface, text_rect)
                
                y += 30
            
            pygame.display.flip()
            self.clock.tick(60)
    
    def draw_menu(self):
        """Draw the current menu"""
        self.screen.fill(self.colors['black'])
        
        # Draw title
        title = "Hand Gesture Car Control"
        title_surface = self.font_large.render(title, True, self.colors['white'])
        title_rect = title_surface.get_rect(center=(self.screen_width // 2, 100))
        self.screen.blit(title_surface, title_rect)
        
        # Draw menu options
        options = self.menu_options[self.current_menu]
        start_y = 250
        self.menu_rects = []  # איפוס רשימת המלבנים
        
        for i, option in enumerate(options):
            # בחירת צבע ופונט על סמך בחירה/hover
            if i == self.selected_option:
                color = self.colors['yellow']
                font = self.font_medium
                
                # רקע לאפשרות הנבחרת
                option_surface = font.render(option, True, color)
                option_rect = option_surface.get_rect(center=(self.screen_width // 2, start_y + i * 60))
                
                # צור מלבן רקע גדול יותר
                background_rect = pygame.Rect(
                    option_rect.left - 20,
                    option_rect.top - 10,
                    option_rect.width + 40,
                    option_rect.height + 20
                )
                pygame.draw.rect(self.screen, self.colors['hover'], background_rect, 0, 10)
                pygame.draw.rect(self.screen, self.colors['yellow'], background_rect, 2, 10)
                
            else:
                color = self.colors['white']
                font = self.font_small
                option_surface = font.render(option, True, color)
                option_rect = option_surface.get_rect(center=(self.screen_width // 2, start_y + i * 60))
            
            # שמירת המלבן עבור זיהוי עכבר
            # צור מלבן גדול יותר לזיהוי עכבר נוח יותר
            mouse_rect = pygame.Rect(
                option_rect.left - 50,
                option_rect.top - 15,
                option_rect.width + 100,
                option_rect.height + 30
            )
            self.menu_rects.append(mouse_rect)
            
            # ציור הטקסט
            self.screen.blit(option_surface, option_rect)
        
        # הוספת הוראות שימוש
        instructions_text = "Use UP/DOWN arrows, Enter, or mouse to select"
        instructions_surface = pygame.font.Font(None, 24).render(instructions_text, True, self.colors['white'])
        instructions_rect = instructions_surface.get_rect(center=(self.screen_width // 2, self.screen_height - 50))
        self.screen.blit(instructions_surface, instructions_rect)
    
    def run(self):
        """Run the game launcher"""
        print("🎮 Game Launcher Started")
        
        while self.running:
            self.handle_events()
            self.draw_menu()
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        print("👋 Game Launcher Ended")
    
    def cleanup(self):
        """Cleanup resources"""
        pygame.quit()

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