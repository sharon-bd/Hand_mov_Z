#!/usr/bin/env python
"""
Renderer Module for Hand Gesture Car Control Game

This module handles rendering the game elements to the screen.
"""

import pygame
import math
import random
import numpy as np
from game.moving_road import MovingRoadGenerator  # Import the MovingRoadGenerator class

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
        
        # Particle systems
        self.particles = []
        
        # Pre-render some common elements
        self._init_prerendered_elements()
        
        # Create the moving road generator
        self.moving_road = MovingRoadGenerator(screen_width, screen_height)
    
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
        רנדור המשחק כולו
        
        Args:
            screen: משטח Pygame לרנדור
            game_state: מצב המשחק הנוכחי
        """
        # קבלת אלמנטים ממצב המשחק
        car = game_state.get('car')
        obstacles = game_state.get('obstacles', [])
        power_ups = game_state.get('power_ups', [])
        score = game_state.get('score', 0)
        
        # וודא שהעולם ממורכז סביב המכונית
        world_offset_x = game_state.get('world_offset_x', 0)
        world_offset_y = game_state.get('world_offset_y', 0)
        
        # ניקוי המסך
        screen.fill(self.colors['black'])
        
        # שימוש במסלול הנע
        if car:
            car_rotation = getattr(car, 'rotation', 0.0)
            car_speed = getattr(car, 'speed', 0.0)
            # עדכון ורנדור המסלול הנע
            self.moving_road.update(car_rotation, car_speed, game_state.get('dt', 0.016))
            self.moving_road.draw(screen, world_offset_x, world_offset_y)
        else:
            # חזרה למסלול סטטי אם אין מכונית
            self._draw_scrolling_road(screen, game_state.get('scroll_speed', self.scroll_speed))
        
        # ציור כל אלמנטי המשחק
        self._draw_obstacles(screen, obstacles)
        self._draw_power_ups(screen, power_ups)
        
        if car:
            self._draw_car(screen, car)
        
        # ציור חלקיקים
        self._update_and_draw_particles(screen, game_state.get('dt', 0.016))
        
        # ציור אלמנטי HUD
        self._draw_hud(screen, game_state)
    
    def _draw_scrolling_road(self, screen, scroll_speed):
        """Draw the road with scrolling lane markings"""
        # Update scroll position
        self.scroll_y += scroll_speed * 0.016  # Assuming 60 FPS
        if self.scroll_y >= self.line_spacing:
            self.scroll_y %= self.line_spacing
        
        # Draw the road background
        screen.blit(self.road_background, (0, 0))
        
        # Draw lane markings
        for lane in range(1, 3):  # 2 lane lines for 3 lanes
            lane_x = self.road_x + lane * self.lane_width
            
            # Draw dashed lines for lanes
            y = -self.line_length + self.scroll_y
            while y < self.screen_height:
                pygame.draw.rect(
                    screen,
                    self.colors['lane'],
                    (lane_x - self.line_width // 2, y, self.line_width, self.line_length)
                )
                y += self.line_spacing
    
    def _draw_car(self, screen, car):
        """Draw the player's car"""
        if hasattr(car, 'draw'):
            # If car has its own draw method, use it
            car.draw(screen)
        else:
            # Fallback to simple rectangle
            car_rect = pygame.Rect(
                car['x'] - car['width'] // 2,
                car['y'] - car['height'] // 2,
                car['width'],
                car['height']
            )
            pygame.draw.rect(screen, self.colors['blue'], car_rect)
    
    def _draw_obstacles(self, screen, obstacles):
        """Draw all obstacles"""
        for obstacle in obstacles:
            self._draw_obstacle(screen, obstacle)
    
    def _draw_obstacle(self, screen, obstacle):
        """Draw a single obstacle"""
        color = self.colors['red'] if obstacle.get('hit', False) else obstacle.get('color', self.colors['gray'])
        
        if obstacle['type'] == 'cone':
            # Draw a triangle for cones
            points = [
                (obstacle['x'], obstacle['y'] - obstacle['height'] // 2),
                (obstacle['x'] - obstacle['width'] // 2, obstacle['y'] + obstacle['height'] // 2),
                (obstacle['x'] + obstacle['width'] // 2, obstacle['y'] + obstacle['height'] // 2)
            ]
            pygame.draw.polygon(screen, color, points)
            
            # Add a white stripe
            pygame.draw.line(
                screen,
                self.colors['white'],
                (obstacle['x'] - obstacle['width'] // 4, obstacle['y']),
                (obstacle['x'] + obstacle['width'] // 4, obstacle['y']),
                2
            )
        elif obstacle['type'] == 'barrier':
            # Draw a rectangle with stripes for barriers
            obstacle_rect = pygame.Rect(
                obstacle['x'] - obstacle['width'] // 2,
                obstacle['y'] - obstacle['height'] // 2,
                obstacle['width'],
                obstacle['height']
            )
            pygame.draw.rect(screen, color, obstacle_rect)
            
            # Add alternating white/red stripes
            stripe_width = obstacle['width'] // 4
            for i in range(4):
                stripe_color = self.colors['white'] if i % 2 == 0 else self.colors['red']
                stripe_rect = pygame.Rect(
                    obstacle['x'] - obstacle['width'] // 2 + i * stripe_width,
                    obstacle['y'] - obstacle['height'] // 2,
                    stripe_width,
                    obstacle['height']
                )
                pygame.draw.rect(screen, stripe_color, stripe_rect)
        else:  # rock or default
            # Draw a circle for rocks/generic obstacles
            pygame.draw.circle(
                screen,
                color,
                (int(obstacle['x']), int(obstacle['y'])),
                obstacle['width'] // 2
            )
            
            # Add some texture
            pygame.draw.circle(
                screen,
                (max(0, color[0] - 30), max(0, color[1] - 30), max(0, color[2] - 30)),
                (int(obstacle['x'] + 5), int(obstacle['y'] - 5)),
                obstacle['width'] // 4
            )
    
    def _draw_power_ups(self, screen, power_ups):
        """Draw all power-ups"""
        for power_up in power_ups:
            self._draw_power_up(screen, power_up)
    
    def _draw_power_up(self, screen, power_up):
        """Draw a single power-up"""
        # Get position with animation offset
        x = int(power_up['x'])
        y = int(power_up['y'] + power_up.get('animation_offset', 0))
        
        # Base shape is a circle with glow
        radius = power_up['width'] // 2
        
        # Draw glow effect
        for r in range(radius, radius - 4, -1):
            alpha = (radius - r) * 60
            s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*power_up['color'], alpha), (r, r), r)
            screen.blit(s, (x - r, y - r))
        
        # Main circle
        pygame.draw.circle(
            screen,
            power_up['color'],
            (x, y),
            radius
        )
        
        # Inner highlight
        pygame.draw.circle(
            screen,
            self.colors['white'],
            (x, y),
            radius // 3
        )
        
        # Draw icon based on power-up type
        if power_up['type'] == 'boost':
            # Draw lightning bolt
            points = [
                (x, y - 10),
                (x - 5, y - 3),
                (x, y),
                (x - 5, y + 10),
                (x, y + 3),
                (x + 5, y - 3)
            ]
            pygame.draw.polygon(screen, self.colors['black'], points)
        elif power_up['type'] == 'shield':
            # Draw shield
            pygame.draw.arc(
                screen,
                self.colors['black'],
                (x - 8, y - 8, 16, 16),
                math.pi / 4,
                math.pi * 7 / 4,
                2
            )
        elif power_up['type'] == 'repair':
            # Draw plus sign
            pygame.draw.line(screen, self.colors['black'], (x - 6, y), (x + 6, y), 2)
            pygame.draw.line(screen, self.colors['black'], (x, y - 6), (x, y + 6), 2)
    
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
        score_text = self.font_medium.render(f"Score: {score}", True, self.colors['white'])
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
    
    def create_explosion(self, x, y, color=(255, 165, 0), num_particles=30):
        """Create an explosion particle effect"""
        for _ in range(num_particles):
            angle = random.uniform(0, math.pi * 2)
            speed = random.uniform(50, 200)
            size = random.randint(2, 8)
            lifetime = random.uniform(0.5, 1.5)
            
            particle = {
                'x': x,
                'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'size': size,
                'color': color,
                'lifetime': lifetime,
                'remaining': lifetime
            }
            self.particles.append(particle)
    
    def create_dust_trail(self, x, y, color=(150, 150, 150), num_particles=5):
        """Create a dust trail particle effect"""
        for _ in range(num_particles):
            angle = random.uniform(math.pi * 0.75, math.pi * 1.25)  # Mostly upward
            speed = random.uniform(20, 60)
            size = random.randint(1, 5)
            lifetime = random.uniform(0.2, 0.8)
            
            particle = {
                'x': x + random.uniform(-10, 10),
                'y': y + random.uniform(-5, 5),
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'size': size,
                'color': color,
                'lifetime': lifetime,
                'remaining': lifetime,
                'fade': True
            }
            self.particles.append(particle)
    
    def _update_and_draw_particles(self, screen, dt):
        """Update and draw all particles"""
        active_particles = []
        
        for particle in self.particles:
            # Update position
            particle['x'] += particle['vx'] * dt
            particle['y'] += particle['vy'] * dt
            
            # Update lifetime
            particle['remaining'] -= dt
            
            # Skip if expired
            if particle['remaining'] <= 0:
                continue
            
            # Apply drag
            particle['vx'] *= 0.95
            particle['vy'] *= 0.95
            
            # Calculate alpha based on remaining lifetime
            alpha = 255
            if particle.get('fade', False):
                alpha = int(255 * particle['remaining'] / particle['lifetime'])
            
            # Calculate size (shrink as it ages)
            size = particle['size']
            if particle.get('shrink', False):
                size = int(particle['size'] * particle['remaining'] / particle['lifetime'])
                size = max(1, size)
            
            # Draw particle
            color = particle['color']
            if alpha < 255:
                s = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*color, alpha), (size, size), size)
                screen.blit(s, (int(particle['x'] - size), int(particle['y'] - size)))
            else:
                pygame.draw.circle(
                    screen,
                    color,
                    (int(particle['x']), int(particle['y'])),
                    size
                )
            
            # Keep active particles
            active_particles.append(particle)
        
        # Update particle list
        self.particles = active_particles
    
    def draw_animated_background(self, screen, t):
        """Draw an animated background for menus"""
        # Fill with gradient
        for y in range(self.screen_height):
            # Create a gradient from dark blue to light blue
            color_value = y / self.screen_height
            color = (
                int(0 + color_value * 173),
                int(0 + color_value * 216),
                int(139 + color_value * (230 - 139))
            )
            pygame.draw.line(screen, color, (0, y), (self.screen_width, y))
        
        # Draw animated dots
        offset = (t * 50) % 50
        for x in range(0, self.screen_width + 50, 50):
            for y in range(0, self.screen_height + 50, 50):
                pygame.draw.circle(
                    screen,
                    self.colors['white'],
                    (int(x + offset), int(y + offset)),
                    2
                )
    
    def draw_menu(self, screen, title, menu_items, selected_item=0, t=0):
        """
        Draw a menu screen
        
        Args:
            screen: Pygame surface to render to
            title: Title of the menu
            menu_items: List of menu items
            selected_item: Index of the selected item
            t: Time for animations
        """
        # Draw animated background
        self.draw_animated_background(screen, t)
        
        # Draw title with shadow
        title_text = self.font_large.render(title, True, self.colors['white'])
        title_shadow = self.font_large.render(title, True, self.colors['black'])
        
        screen.blit(title_shadow, (self.screen_width // 2 - title_text.get_width() // 2 + 2, 80 + 2))
        screen.blit(title_text, (self.screen_width // 2 - title_text.get_width() // 2, 80))
        
        # Draw menu items
        y = 200
        for i, item in enumerate(menu_items):
            # Determine color and size based on selection
            if i == selected_item:
                color = self.colors['yellow']
                font = self.font_medium
                # Draw selection box
                item_text = font.render(item, True, color)
                box_rect = pygame.Rect(
                    self.screen_width // 2 - item_text.get_width() // 2 - 20,
                    y - 10,
                    item_text.get_width() + 40,
                    item_text.get_height() + 20
                )
                pygame.draw.rect(screen, (0, 0, 200), box_rect, 0, 10)
                pygame.draw.rect(screen, (100, 100, 255), box_rect, 2, 10)
            else:
                color = self.colors['white']
                font = self.font_small
            
            # Draw the item text
            item_text = font.render(item, True, color)
            screen.blit(item_text, (self.screen_width // 2 - item_text.get_width() // 2, y))
            
            # Space between items
            y += 50
    
    def draw_game_over(self, screen, score, high_score):
        """Draw the game over screen"""
        # Create a semi-transparent overlay
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        screen.blit(overlay, (0, 0))
        
        # Draw "Game Over" text
        game_over_text = self.font_large.render("Game Over", True, self.colors['white'])
        screen.blit(game_over_text, (self.screen_width // 2 - game_over_text.get_width() // 2, 150))
        
        # Draw score
        score_text = self.font_medium.render(f"Your Score: {score}", True, self.colors['white'])
        screen.blit(score_text, (self.screen_width // 2 - score_text.get_width() // 2, 250))
        
        # Draw high score if achieved
        if score >= high_score:
            high_score_text = self.font_medium.render("New High Score!", True, self.colors['yellow'])
            screen.blit(high_score_text, (self.screen_width // 2 - high_score_text.get_width() // 2, 300))
        else:
            high_score_text = self.font_medium.render(f"High Score: {high_score}", True, self.colors['white'])
            screen.blit(high_score_text, (self.screen_width // 2 - high_score_text.get_width() // 2, 300))
        
        # Draw restart instructions
        restart_text = self.font_medium.render("Press SPACE to restart or ESC to quit", True, self.colors['white'])
        screen.blit(restart_text, (self.screen_width // 2 - restart_text.get_width() // 2, 400))
    
    def draw_tutorial(self, screen, step_text, current_step, total_steps):
        """Draw a tutorial overlay with the given text"""
        # Create a semi-transparent overlay
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        screen.blit(overlay, (0, 0))
        
        # Draw title
        title_text = self.font_large.render("Tutorial", True, self.colors['white'])
        screen.blit(title_text, (self.screen_width // 2 - title_text.get_width() // 2, 100))
        
        # Draw step text with word wrap
        words = step_text.split()
        lines = []
        line = []
        line_width = 0
        
        # Word wrap to fit screen width
        for word in words:
            word_surface = self.font_medium.render(word + " ", True, self.colors['white'])
            word_width = word_surface.get_width()
            if line_width + word_width <= self.screen_width - 100:
                line.append(word)
                line_width += word_width
            else:
                lines.append(" ".join(line))
                line = [word]
                line_width = word_width
        
        if line:
            lines.append(" ".join(line))
        
        # Draw each line
        y = 200
        for line in lines:
            line_text = self.font_medium.render(line, True, self.colors['white'])
            screen.blit(line_text, (self.screen_width // 2 - line_text.get_width() // 2, y))
            y += 40
        
        # Draw progress indicators
        for i in range(total_steps):
            color = self.colors['green'] if i == current_step else self.colors['light_gray']
            pygame.draw.circle(
                screen,
                color,
                (self.screen_width // 2 - (total_steps - 1) * 15 + i * 30, 350),
                10
            )
        
        # Draw continue hint
        hint_text = self.font_small.render("Press SPACE to continue", True, self.colors['white'])
        screen.blit(hint_text, (self.screen_width // 2 - hint_text.get_width() // 2, 400))