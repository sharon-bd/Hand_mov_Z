#!/usr/bin/env python
"""
Obstacle Module for Hand Gesture Car Control Game

This module implements various types of obstacles that appear in the game.
"""

import pygame
import random
import math

class Obstacle:
    """
    Represents an obstacle in the game
    """
    
    def __init__(self, x, y, obstacle_type="cone", speed=200):
        """
        Initialize an obstacle
        
        Args:
            x, y: Initial position
            obstacle_type: Type of obstacle ('cone', 'barrier', 'rock', 'puddle')
            speed: Speed at which obstacle moves
        """
        self.x = x
        self.y = y
        self.initial_x = x
        self.initial_y = y
        self.obstacle_type = obstacle_type
        self.speed = speed
        self.hit = False
        self.animation_frame = 0
        self.creation_time = pygame.time.get_ticks()
        
        # Set dimensions and color based on type
        self._setup_obstacle_properties()
        
        # Create collision rectangle
        self.rect = pygame.Rect(
            self.x - self.width // 2,
            self.y - self.height // 2,
            self.width,
            self.height
        )
    
    def _setup_obstacle_properties(self):
        """Setup obstacle properties based on type"""
        if self.obstacle_type == "cone":
            self.width = 30
            self.height = 40
            self.color = (255, 140, 0)  # Orange
            self.damage = 10
        elif self.obstacle_type == "barrier":
            self.width = 80
            self.height = 30
            self.color = (255, 0, 0)  # Red
            self.damage = 20
        elif self.obstacle_type == "rock":
            self.width = 40
            self.height = 40
            self.color = (120, 120, 120)  # Gray
            self.damage = 15
        elif self.obstacle_type == "puddle":
            self.width = 60
            self.height = 60
            self.color = (0, 100, 150)  # Blue
            self.damage = 0  # No damage, just slip effect
        else:
            # Default obstacle
            self.width = 30
            self.height = 30
            self.color = (100, 100, 100)
            self.damage = 5
    
    def update(self, dt):
        """
        Update obstacle position and animation
        
        Args:
            dt: Time delta in seconds
        """
        # Move obstacle down the screen
        self.y += self.speed * dt
        
        # Update collision rectangle
        self.rect.centerx = self.x
        self.rect.centery = self.y
        
        # Update animation frame
        self.animation_frame += dt * 10  # Animation speed
    
    def draw(self, screen):
        """
        Draw the obstacle on the screen
        
        Args:
            screen: Pygame surface to draw on
        """
        # Choose color based on hit state
        color = (255, 0, 0) if self.hit else self.color
        
        if self.obstacle_type == "cone":
            self._draw_cone(screen, color)
        elif self.obstacle_type == "barrier":
            self._draw_barrier(screen, color)
        elif self.obstacle_type == "rock":
            self._draw_rock(screen, color)
        elif self.obstacle_type == "puddle":
            self._draw_puddle(screen, color)
        else:
            self._draw_default(screen, color)
    
    def _draw_cone(self, screen, color):
        """Draw a traffic cone"""
        # Draw cone shape
        points = [
            (self.x, self.y - self.height // 2),
            (self.x - self.width // 2, self.y + self.height // 2),
            (self.x + self.width // 2, self.y + self.height // 2)
        ]
        pygame.draw.polygon(screen, color, points)
        
        # Draw white stripe
        pygame.draw.line(
            screen,
            (255, 255, 255),
            (self.x - self.width // 4, self.y),
            (self.x + self.width // 4, self.y),
            3
        )
        
        # Draw base
        pygame.draw.ellipse(
            screen,
            (50, 50, 50),
            (self.x - self.width // 2, self.y + self.height // 2 - 5, self.width, 10)
        )
    
    def _draw_barrier(self, screen, color):
        """Draw a barrier with stripes"""
        # Draw main barrier
        barrier_rect = pygame.Rect(
            self.x - self.width // 2,
            self.y - self.height // 2,
            self.width,
            self.height
        )
        pygame.draw.rect(screen, color, barrier_rect)
        
        # Draw diagonal stripes
        stripe_width = 10
        for i in range(0, self.width, stripe_width * 2):
            stripe_rect = pygame.Rect(
                self.x - self.width // 2 + i,
                self.y - self.height // 2,
                stripe_width,
                self.height
            )
            pygame.draw.rect(screen, (255, 255, 255), stripe_rect)
    
    def _draw_rock(self, screen, color):
        """Draw a rock obstacle"""
        # Draw main rock circle
        pygame.draw.circle(
            screen,
            color,
            (int(self.x), int(self.y)),
            self.width // 2
        )
        
        # Add texture with smaller circles
        pygame.draw.circle(
            screen,
            (max(0, color[0] - 30), max(0, color[1] - 30), max(0, color[2] - 30)),
            (int(self.x + 5), int(self.y - 5)),
            self.width // 4
        )
        
        pygame.draw.circle(
            screen,
            (min(255, color[0] + 20), min(255, color[1] + 20), min(255, color[2] + 20)),
            (int(self.x - 3), int(self.y + 3)),
            self.width // 6
        )
    
    def _draw_puddle(self, screen, color):
        """Draw a water puddle with animation"""
        # Animate puddle with sine wave
        animation_offset = math.sin(self.animation_frame) * 3
        
        # Draw multiple ellipses for water effect
        for i in range(3):
            alpha = 150 - i * 30
            size_mult = 1 - i * 0.1
            
            # Create surface with alpha
            puddle_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.ellipse(
                puddle_surface,
                (*color, alpha),
                (0, 0, int(self.width * size_mult), int(self.height * size_mult))
            )
            
            screen.blit(
                puddle_surface,
                (self.x - (self.width * size_mult) // 2 + animation_offset,
                 self.y - (self.height * size_mult) // 2)
            )
    
    def _draw_default(self, screen, color):
        """Draw default obstacle as a rectangle"""
        obstacle_rect = pygame.Rect(
            self.x - self.width // 2,
            self.y - self.height // 2,
            self.width,
            self.height
        )
        pygame.draw.rect(screen, color, obstacle_rect)
        pygame.draw.rect(screen, (255, 255, 255), obstacle_rect, 2)
    
    def is_off_screen(self, screen_height):
        """
        Check if obstacle is off screen
        
        Args:
            screen_height: Height of the screen
            
        Returns:
            Boolean indicating if obstacle is off screen
        """
        return self.y > screen_height + self.height
    
    def get_collision_damage(self):
        """
        Get damage this obstacle inflicts on collision
        
        Returns:
            Damage amount
        """
        return self.damage
    
    def get_collision_effect(self):
        """
        Get special effect this obstacle has on collision
        
        Returns:
            String describing the effect
        """
        if self.obstacle_type == "puddle":
            return "slip"
        elif self.obstacle_type == "barrier":
            return "stop"
        else:
            return "damage"

class ObstacleManager:
    """Manages creation and updating of obstacles"""
    
    def __init__(self, screen_width, screen_height):
        """
        Initialize obstacle manager
        
        Args:
            screen_width: Width of game screen
            screen_height: Height of game screen
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.obstacles = []
        self.spawn_timer = 0
        self.spawn_interval = 2.0  # Seconds between spawns
        
        # Obstacle type probabilities
        self.obstacle_types = [
            ("cone", 0.4),
            ("rock", 0.3),
            ("barrier", 0.2),
            ("puddle", 0.1)
        ]
    
    def update(self, dt, difficulty_multiplier=1.0):
        """
        Update all obstacles
        
        Args:
            dt: Time delta in seconds
            difficulty_multiplier: Multiplier for spawn rate
        """
        # Update spawn timer
        self.spawn_timer += dt
        
        # Spawn new obstacles
        spawn_rate = self.spawn_interval / difficulty_multiplier
        if self.spawn_timer >= spawn_rate:
            self.spawn_obstacle()
            self.spawn_timer = 0
        
        # Update existing obstacles
        for obstacle in self.obstacles[:]:
            obstacle.update(dt)
            
            # Remove obstacles that are off screen
            if obstacle.is_off_screen(self.screen_height):
                self.obstacles.remove(obstacle)
    
    def spawn_obstacle(self):
        """Spawn a new obstacle"""
        # Choose random position
        margin = 50
        x = random.randint(margin, self.screen_width - margin)
        y = -50  # Start above screen
        
        # Choose obstacle type based on probabilities
        rand_val = random.random()
        cumulative_prob = 0
        obstacle_type = "cone"  # Default
        
        for obs_type, prob in self.obstacle_types:
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                obstacle_type = obs_type
                break
        
        # Create obstacle
        speed = random.randint(150, 250)
        obstacle = Obstacle(x, y, obstacle_type, speed)
        self.obstacles.append(obstacle)
    
    def draw_all(self, screen):
        """
        Draw all obstacles
        
        Args:
            screen: Pygame surface to draw on
        """
        for obstacle in self.obstacles:
            obstacle.draw(screen)
    
    def check_collisions(self, car_rect):
        """
        Check collisions with car
        
        Args:
            car_rect: Car's collision rectangle
            
        Returns:
            List of obstacles that collided with car
        """
        collisions = []
        for obstacle in self.obstacles:
            if not obstacle.hit and obstacle.rect.colliderect(car_rect):
                obstacle.hit = True
                collisions.append(obstacle)
        
        return collisions
    
    def clear_all(self):
        """Clear all obstacles"""
        self.obstacles.clear()
    
    def get_obstacle_count(self):
        """Get number of active obstacles"""
        return len(self.obstacles)
