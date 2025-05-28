#!/usr/bin/env python
"""
Obstacle Module for Hand Gesture Car Control Game

This module implements various types of obstacles and their management.
"""

import pygame
import random
import math
import time

class Obstacle:
    """Base obstacle class"""
    
    def __init__(self, x, y, obstacle_type="cone", speed=200):
        """
        Initialize an obstacle
        
        Args:
            x, y: Position coordinates
            obstacle_type: Type of obstacle ("cone", "barrier", "rock", "puddle")
            speed: Movement speed
        """
        self.x = x
        self.y = y
        self.original_x = x
        self.original_y = y
        self.type = obstacle_type
        self.speed = speed
        self.hit = False
        
        # Set size and color based on type
        if obstacle_type == "cone":
            self.width = 30
            self.height = 40
            self.color = (255, 140, 0)  # Orange
        elif obstacle_type == "barrier":
            self.width = 80
            self.height = 30
            self.color = (255, 0, 0)  # Red
        elif obstacle_type == "rock":
            self.width = 40
            self.height = 40
            self.color = (120, 120, 120)  # Gray
        elif obstacle_type == "puddle":
            self.width = 60
            self.height = 40
            self.color = (0, 100, 200)  # Blue
        else:
            self.width = 30
            self.height = 30
            self.color = (100, 100, 100)  # Default gray
        
        # Create collision rectangle
        self.rect = pygame.Rect(
            self.x - self.width // 2,
            self.y - self.height // 2,
            self.width,
            self.height
        )
        
        # Animation properties
        self.animation_time = 0
        self.flash_timer = 0
        
    def update(self, dt):
        """Update obstacle state"""
        # Move obstacle
        self.y += self.speed * dt
        
        # Update collision rectangle
        self.rect.x = self.x - self.width // 2
        self.rect.y = self.y - self.height // 2
        
        # Update animation
        self.animation_time += dt
        
        # Flash effect when hit
        if self.hit:
            self.flash_timer += dt
    
    def draw(self, screen):
        """Draw the obstacle"""
        # Determine color based on state
        color = self.color
        if self.hit:
            # Flash red when hit
            if int(self.flash_timer * 10) % 2 == 0:
                color = (255, 0, 0)
        
        if self.type == "cone":
            self._draw_cone(screen, color)
        elif self.type == "barrier":
            self._draw_barrier(screen, color)
        elif self.type == "rock":
            self._draw_rock(screen, color)
        elif self.type == "puddle":
            self._draw_puddle(screen, color)
        else:
            # Default rectangle
            pygame.draw.rect(screen, color, self.rect)
    
    def _draw_cone(self, screen, color):
        """Draw a traffic cone"""
        # Main triangle
        points = [
            (self.x, self.y - self.height // 2),
            (self.x - self.width // 2, self.y + self.height // 2),
            (self.x + self.width // 2, self.y + self.height // 2)
        ]
        pygame.draw.polygon(screen, color, points)
        
        # White stripes
        stripe_y = self.y - self.height // 4
        pygame.draw.line(
            screen,
            (255, 255, 255),
            (self.x - self.width // 3, stripe_y),
            (self.x + self.width // 3, stripe_y),
            3
        )
    
    def _draw_barrier(self, screen, color):
        """Draw a barrier"""
        # Main rectangle
        pygame.draw.rect(screen, color, self.rect)
        
        # Diagonal stripes
        stripe_width = self.width // 8
        for i in range(8):
            stripe_color = (255, 255, 255) if i % 2 == 0 else color
            stripe_rect = pygame.Rect(
                self.rect.x + i * stripe_width,
                self.rect.y,
                stripe_width,
                self.rect.height
            )
            pygame.draw.rect(screen, stripe_color, stripe_rect)
    
    def _draw_rock(self, screen, color):
        """Draw a rock"""
        # Main circle
        pygame.draw.circle(
            screen,
            color,
            (self.x, self.y),
            self.width // 2
        )
        
        # Add texture
        darker_color = (
            max(0, color[0] - 30),
            max(0, color[1] - 30),
            max(0, color[2] - 30)
        )
        pygame.draw.circle(
            screen,
            darker_color,
            (self.x + 5, self.y - 5),
            self.width // 4
        )
    
    def _draw_puddle(self, screen, color):
        """Draw a water puddle"""
        # Elliptical shape
        puddle_rect = pygame.Rect(
            self.x - self.width // 2,
            self.y - self.height // 2,
            self.width,
            self.height
        )
        pygame.draw.ellipse(screen, color, puddle_rect)
        
        # Ripple effect
        ripple_radius = int(10 + 5 * math.sin(self.animation_time * 4))
        pygame.draw.circle(
            screen,
            (0, 150, 255),
            (self.x, self.y),
            ripple_radius,
            2
        )
    
    def is_off_screen(self, screen_height):
        """Check if obstacle is off screen"""
        return self.y > screen_height + 50

class ObstacleManager:
    """Manager for all obstacles in the game"""
    
    def __init__(self, screen_width, screen_height):
        """
        Initialize the obstacle manager
        
        Args:
            screen_width: Width of the game screen
            screen_height: Height of the game screen
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.obstacles = []
        
        # Spawn parameters
        self.spawn_timer = 0
        self.spawn_interval = 2.0  # Seconds between spawns
        self.last_spawn_time = 0
        
        # Difficulty scaling
        self.difficulty_factor = 1.0
        self.max_obstacles = 10
        
        # Obstacle types and their spawn weights
        self.obstacle_types = {
            "cone": 0.4,      # 40% chance
            "barrier": 0.3,   # 30% chance
            "rock": 0.2,      # 20% chance
            "puddle": 0.1     # 10% chance
        }
    
    def update(self, dt, car_speed=1.0):
        """
        Update all obstacles
        
        Args:
            dt: Delta time
            car_speed: Current car speed (affects obstacle spawn rate)
        """
        # Update spawn timer
        self.spawn_timer += dt
        
        # Adjust difficulty based on car speed
        self.difficulty_factor = 1.0 + car_speed * 0.5
        
        # Check if we should spawn a new obstacle
        adjusted_interval = max(0.5, self.spawn_interval / self.difficulty_factor)
        if (self.spawn_timer >= adjusted_interval and 
            len(self.obstacles) < self.max_obstacles):
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
        # Choose obstacle type based on weights
        obstacle_type = self._choose_obstacle_type()
        
        # Choose spawn position
        road_center = self.screen_width // 2
        road_width = 300
        spawn_x = random.randint(
            road_center - road_width // 2 + 50,
            road_center + road_width // 2 - 50
        )
        spawn_y = -50  # Above screen
        
        # Create obstacle with adjusted speed
        base_speed = 200
        speed = base_speed * self.difficulty_factor
        
        obstacle = Obstacle(spawn_x, spawn_y, obstacle_type, speed)
        self.obstacles.append(obstacle)
    
    def _choose_obstacle_type(self):
        """Choose obstacle type based on weights"""
        rand = random.random()
        cumulative = 0
        
        for obstacle_type, weight in self.obstacle_types.items():
            cumulative += weight
            if rand <= cumulative:
                return obstacle_type
        
        return "cone"  # Fallback
    
    def draw(self, screen):
        """Draw all obstacles"""
        for obstacle in self.obstacles:
            obstacle.draw(screen)
    
    def check_collision(self, car_rect):
        """
        Check collision with car
        
        Args:
            car_rect: Car's collision rectangle
            
        Returns:
            Obstacle that was hit, or None
        """
        for obstacle in self.obstacles:
            if not obstacle.hit and obstacle.rect.colliderect(car_rect):
                obstacle.hit = True
                return obstacle
        
        return None
    
    def clear(self):
        """Clear all obstacles"""
        self.obstacles.clear()
        self.spawn_timer = 0
    
    def get_obstacle_count(self):
        """Get current number of obstacles"""
        return len(self.obstacles)
    
    def set_difficulty(self, difficulty):
        """
        Set difficulty level
        
        Args:
            difficulty: Difficulty multiplier (1.0 = normal)
        """
        self.difficulty_factor = difficulty
        self.spawn_interval = max(0.3, 2.0 / difficulty)
        self.max_obstacles = min(20, int(10 * difficulty))

class PowerUp:
    """Power-up class"""
    
    def __init__(self, x, y, power_type="boost"):
        """
        Initialize a power-up
        
        Args:
            x, y: Position coordinates
            power_type: Type of power-up ("boost", "shield", "repair", "slow_time")
        """
        self.x = x
        self.y = y
        self.type = power_type
        self.collected = False
        self.animation_time = 0
        
        # Set properties based on type
        if power_type == "boost":
            self.color = (255, 215, 0)  # Gold
            self.effect_duration = 3.0
        elif power_type == "shield":
            self.color = (0, 255, 255)  # Cyan
            self.effect_duration = 5.0
        elif power_type == "repair":
            self.color = (0, 255, 0)  # Green
            self.effect_duration = 0  # Instant
        elif power_type == "slow_time":
            self.color = (255, 0, 255)  # Magenta
            self.effect_duration = 4.0
        
        self.width = 30
        self.height = 30
        self.speed = 150
        
        # Create collision rectangle
        self.rect = pygame.Rect(
            self.x - self.width // 2,
            self.y - self.height // 2,
            self.width,
            self.height
        )
    
    def update(self, dt):
        """Update power-up state"""
        if not self.collected:
            # Move down
            self.y += self.speed * dt
            
            # Update collision rectangle
            self.rect.x = self.x - self.width // 2
            self.rect.y = self.y - self.height // 2
            
            # Update animation
            self.animation_time += dt
    
    def draw(self, screen):
        """Draw the power-up"""
        if self.collected:
            return
        
        # Pulsing effect
        pulse = 1.0 + 0.3 * math.sin(self.animation_time * 6)
        radius = int((self.width // 2) * pulse)
        
        # Glow effect
        for r in range(radius, radius - 8, -2):
            alpha = max(0, 100 - (radius - r) * 20)
            s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            color_with_alpha = (*self.color, alpha)
            pygame.draw.circle(s, color_with_alpha, (r, r), r)
            screen.blit(s, (self.x - r, self.y - r))
        
        # Main circle
        pygame.draw.circle(
            screen,
            self.color,
            (self.x, self.y),
            radius
        )
        
        # Icon based on type
        if self.type == "boost":
            # Lightning bolt
            points = [
                (self.x - 8, self.y - 10),
                (self.x - 3, self.y - 2),
                (self.x + 3, self.y - 2),
                (self.x + 8, self.y + 10),
                (self.x + 3, self.y + 2),
                (self.x - 3, self.y + 2)
            ]
            pygame.draw.polygon(screen, (0, 0, 0), points)
        elif self.type == "shield":
            # Shield shape
            shield_points = [
                (self.x, self.y - 8),
                (self.x - 6, self.y - 4),
                (self.x - 6, self.y + 4),
                (self.x, self.y + 8),
                (self.x + 6, self.y + 4),
                (self.x + 6, self.y - 4)
            ]
            pygame.draw.polygon(screen, (0, 0, 0), shield_points)
        elif self.type == "repair":
            # Plus sign
            pygame.draw.line(screen, (0, 0, 0), (self.x - 6, self.y), (self.x + 6, self.y), 3)
            pygame.draw.line(screen, (0, 0, 0), (self.x, self.y - 6), (self.x, self.y + 6), 3)
        elif self.type == "slow_time":
            # Clock icon
            pygame.draw.circle(screen, (0, 0, 0), (self.x, self.y), 8, 2)
            pygame.draw.line(screen, (0, 0, 0), (self.x, self.y), (self.x, self.y - 6), 2)
            pygame.draw.line(screen, (0, 0, 0), (self.x, self.y), (self.x + 4, self.y), 2)
    
    def is_off_screen(self, screen_height):
        """Check if power-up is off screen"""
        return self.y > screen_height + 50

class PowerUpManager:
    """Manager for all power-ups in the game"""
    
    def __init__(self, screen_width, screen_height):
        """
        Initialize the power-up manager
        
        Args:
            screen_width: Width of the game screen
            screen_height: Height of the game screen
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.power_ups = []
        
        # Spawn parameters
        self.spawn_timer = 0
        self.spawn_interval = 8.0  # Seconds between spawns
        
        # Power-up types and their spawn weights
        self.power_up_types = {
            "boost": 0.4,      # 40% chance
            "shield": 0.3,     # 30% chance
            "repair": 0.2,     # 20% chance
            "slow_time": 0.1   # 10% chance
        }
    
    def update(self, dt):
        """Update all power-ups"""
        # Update spawn timer
        self.spawn_timer += dt
        
        # Check if we should spawn a new power-up
        if self.spawn_timer >= self.spawn_interval:
            if random.random() < 0.3:  # 30% chance to spawn
                self.spawn_power_up()
            self.spawn_timer = 0
        
        # Update existing power-ups
        for power_up in self.power_ups[:]:
            power_up.update(dt)
            
            # Remove power-ups that are off screen or collected
            if power_up.is_off_screen(self.screen_height) or power_up.collected:
                self.power_ups.remove(power_up)
    
    def spawn_power_up(self):
        """Spawn a new power-up"""
        # Choose power-up type
        power_type = self._choose_power_up_type()
        
        # Choose spawn position
        road_center = self.screen_width // 2
        road_width = 300
        spawn_x = random.randint(
            road_center - road_width // 2 + 50,
            road_center + road_width // 2 - 50
        )
        spawn_y = -50  # Above screen
        
        power_up = PowerUp(spawn_x, spawn_y, power_type)
        self.power_ups.append(power_up)
    
    def _choose_power_up_type(self):
        """Choose power-up type based on weights"""
        rand = random.random()
        cumulative = 0
        
        for power_type, weight in self.power_up_types.items():
            cumulative += weight
            if rand <= cumulative:
                return power_type
        
        return "boost"  # Fallback
    
    def draw(self, screen):
        """Draw all power-ups"""
        for power_up in self.power_ups:
            power_up.draw(screen)
    
    def check_collision(self, car_rect):
        """
        Check collision with car
        
        Args:
            car_rect: Car's collision rectangle
            
        Returns:
            Power-up that was collected, or None
        """
        for power_up in self.power_ups:
            if not power_up.collected and power_up.rect.colliderect(car_rect):
                power_up.collected = True
                return power_up
        
        return None
    
    def clear(self):
        """Clear all power-ups"""
        self.power_ups.clear()
        self.spawn_timer = 0
    
    def get_power_up_count(self):
        """Get current number of power-ups"""
        return len([p for p in self.power_ups if not p.collected])
