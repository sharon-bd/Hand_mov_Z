#!/usr/bin/env python
"""
Renderer Module for Hand Gesture Car Control Game

This module handles rendering the game elements to the screen.
"""

import pygame
import math
import random

class GameRenderer:
    """Handles rendering for the game"""
    
    def __init__(self, screen_width, screen_height):
        """
        Initialize the renderer
        
        Args:
            screen_width: Width of the game screen
            screen_height: Height of the game screen
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Colors
        self.colors = {
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'gray': (100, 100, 100),
            'light_gray': (200, 200, 200),
            'dark_gray': (50, 50, 50),
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'road': (80, 80, 80),
            'lane': (240, 240, 240),
            'grass': (0, 100, 0)
        }
        
        # Road parameters
        self.road_width = 600
        self.road_x = (screen_width - self.road_width) // 2
        self.lane_width = self.road_width // 3
        
        # Scrolling parameters
        self.scroll_y = 0
        self.scroll_speed = 500  # pixels per second
        
        # Line parameters
        self.line_spacing = 50
        self.line_length = 30
        self.line_width = 10
        
        # Pre-render some common elements
        self._init_prerendered_elements()
        
        # Create the simple moving road generator
        try:
            from .moving_road import MovingRoadGenerator
            self.moving_road = MovingRoadGenerator(screen_width, screen_height)
        except ImportError:
            self.moving_road = None
    
    def _init_prerendered_elements(self):
        """Pre-render common elements for performance"""
        # Pre-render the road background
        self.road_background = pygame.Surface((self.screen_width, self.screen_height))
        self.road_background.fill(self.colors['grass'])
        pygame.draw.rect(
            self.road_background,
            self.colors['road'],
            (self.road_x, 0, self.road_width, self.screen_height)
        )
    
    def render_game(self, screen, game_state):
        """
        Render the complete game
        
        Args:
            screen: Pygame surface to render to
            game_state: Current game state
        """
        # Clear screen
        screen.fill(self.colors['black'])
        
        # Use simple moving road
        if self.moving_road and (car := game_state.get('car')):
            car_rotation = getattr(car, 'rotation', 0.0)
            car_speed = getattr(car, 'speed', 0.0)
            # Update and render the simple moving road
            self.moving_road.update(car_rotation, car_speed, game_state.get('dt', 0.016))
            self.moving_road.draw(screen)
        else:
            # Fallback to static road
            self._draw_static_road(screen)
        
        # Draw game objects
        self._draw_obstacles(screen, game_state.get('obstacles', []))
        
        if car := game_state.get('car'):
            self._draw_car(screen, car)
        
        # Draw HUD elements
        self._draw_hud(screen, game_state)
    
    def _draw_static_road(self, screen):
        """Draw simple static road with moving center line"""
        # Draw grass background
        screen.fill(self.colors['grass'])
        
        # Draw road surface
        road_rect = pygame.Rect(
            self.screen_width // 2 - 300 // 2,
            0,
            300,
            self.screen_height
        )
        pygame.draw.rect(screen, self.colors['road'], road_rect)
        
        # Draw moving center line that matches obstacle movement
        center_x = self.screen_width // 2
        line_width = 6
        dash_length = 30
        gap_length = 20
        dash_spacing = dash_length + gap_length
        
        # Use negative scroll to match cone direction
        offset = int(-self.scroll_y) % dash_spacing
        
        y = -offset
        while y < self.screen_height + dash_spacing:
            if y + dash_length > 0 and y < self.screen_height:
                dash_start = max(0, y)
                dash_end = min(self.screen_height, y + dash_length)
                
                pygame.draw.rect(screen, self.colors['lane'],
                               (center_x - line_width // 2, dash_start,
                                line_width, dash_end - dash_start))
            
            y += dash_spacing

    
    def _draw_obstacles(self, screen, obstacles):
        """Draw all obstacles"""
        for obstacle in obstacles:
            self._draw_obstacle(screen, obstacle)
    
    def _draw_obstacle(self, screen, obstacle):
        """Draw a single obstacle"""
        color = self.colors['red'] if obstacle.get('hit', False) else obstacle.get('color', self.colors['gray'])
        
        if hasattr(obstacle, 'draw'):
            obstacle.draw(screen)
        else:
            # Fallback drawing
            pygame.draw.rect(screen, color, 
                           (obstacle['x'] - obstacle['width'] // 2,
                            obstacle['y'] - obstacle['height'] // 2,
                            obstacle['width'], obstacle['height']))
    
    def _draw_car(self, screen, car):
        """Draw the player's car"""
        if hasattr(car, 'draw'):
            # If car has its own draw method, use it
            car.draw(screen)
        else:
            # Fallback to simple rectangle
            car_rect = pygame.Rect(
                car.x - car.width // 2,
                car.y - car.height // 2,
                car.width,
                car.height
            )
            pygame.draw.rect(screen, self.colors['blue'], car_rect)
    
    def _draw_hud(self, screen, game_state):
        """Draw the Heads-Up Display elements"""
        score = game_state.get('score', 0)
        health = game_state.get('health', 100)
        time_left = game_state.get('time_left')
        
        # Create fonts if needed
        if not hasattr(self, 'font_large'):
            self.font_large = pygame.font.Font(None, 48)
            self.font_medium = pygame.font.Font(None, 36)
            self.font_small = pygame.font.Font(None, 24)
        
        # Draw score
        score_text = self.font_medium.render(f"Score: {int(score)}", True, self.colors['white'])
        screen.blit(score_text, (20, 20))
        
        # Draw health bar
        health_width = 200
        health_height = 20
        health_x = self.screen_width - health_width - 20
        health_y = 20
        
        # Health bar background
        pygame.draw.rect(
            screen,
            self.colors['dark_gray'],
            (health_x, health_y, health_width, health_height)
        )
        
        # Health bar fill
        health_fill_width = int(health_width * health / 100)
        health_color = self.colors['green']
        if health < 50:
            health_color = self.colors['yellow']
        if health < 25:
            health_color = self.colors['red']
            
        pygame.draw.rect(
            screen,
            health_color,
            (health_x, health_y, health_fill_width, health_height)
        )
        
        # Health text
        health_text = self.font_small.render(f"Health: {health}%", True, self.colors['white'])
        screen.blit(health_text, (health_x + health_width // 2 - health_text.get_width() // 2, health_y + 25))
        
        # Draw time left if applicable
        if time_left is not None:
            minutes = int(time_left) // 60
            seconds = int(time_left) % 60
            time_text = self.font_medium.render(f"Time: {minutes}:{seconds:02d}", True, self.colors['white'])
            screen.blit(time_text, (self.screen_width // 2 - time_text.get_width() // 2, 20))