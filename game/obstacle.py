#!/usr/bin/env python
"""
Obstacle Module for Hand Gesture Car Control Game

This module implements various types of obstacles that appear in the game.
"""

import pygame
import random
import math

class Obstacle:
    """Represents an obstacle in the game"""
    
    def __init__(self, x, y, obstacle_type="cone", speed=200):
        """Initialize an obstacle"""
        self.x = x
        self.y = y
        self.world_x = x  # Position in world coordinates
        self.obstacle_type = obstacle_type
        self.speed = speed
        self.hit = False
        
        # Set dimensions and color based on type
        if obstacle_type == "cone":
            self.width = 30
            self.height = 40
            self.color = (255, 140, 0)
            self.damage = 10
        elif obstacle_type == "barrier":
            self.width = 80
            self.height = 30
            self.color = (255, 0, 0)
            self.damage = 20
        elif obstacle_type == "rock":
            self.width = 40
            self.height = 40
            self.color = (120, 120, 120)
            self.damage = 15
        else:
            self.width = 30
            self.height = 30
            self.color = (100, 100, 100)
            self.damage = 5
        
        # Create collision rectangle
        self.rect = pygame.Rect(
            self.x - self.width // 2,
            self.y - self.height // 2,
            self.width,
            self.height
        )

    def update(self, dt):
        """Update obstacle position - moving DOWN the screen"""
        self.y += self.speed * dt  # POSITIVE direction = moving down
        self.rect.centerx = self.x
        self.rect.centery = self.y

    def draw(self, screen, world_offset_x=0):
        """Draw the obstacle with minimal world offset effect"""
        # FIXED: Minimal screen position change - mostly stay in place
        screen_x = self.x - (world_offset_x * 0.05)  # Only 5% of offset effect
        
        # Only draw if visible on screen
        if -50 <= screen_x <= screen.get_width() + 50:
            color = (255, 0, 0) if self.hit else self.color
            
            # Draw cone shape
            points = [
                (screen_x, self.y - self.height // 2),
                (screen_x - self.width // 2, self.y + self.height // 2),
                (screen_x + self.width // 2, self.y + self.height // 2)
            ]
            pygame.draw.polygon(screen, color, points)
            
            # White stripe
            pygame.draw.line(screen, (255, 255, 255),
                            (screen_x - self.width // 4, self.y),
                            (screen_x + self.width // 4, self.y), 3)

class ObstacleManager:
    """Manages creation and updating of obstacles"""
    
    def __init__(self):
        """Initialize obstacle manager with default parameters"""
        self.obstacles = []
        self.spawn_rate = 0.02
        self.obstacle_speed = 200
        
        # Default road parameters - will be updated when needed
        self.screen_width = 800
        self.screen_height = 600
        self.road_width = 600
        self.road_x = 100
        
        # World offset for steering effect - MINIMAL
        self.world_offset_x = 0

    def set_road_parameters(self, screen_width, screen_height, road_width, road_x):
        """Set road parameters after initialization"""
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.road_width = road_width
        self.road_x = road_x

    def update_world_offset(self, offset_x):
        """Update world offset for steering effect - MINIMAL IMPACT"""
        # FIXED: Reduce the offset effect to almost nothing
        self.world_offset_x = offset_x * 0.1  # Only 10% of original effect

    def update(self, dt, score):
        """Update all obstacles with minimal steering effect"""
        active_obstacles = []
        for obstacle in self.obstacles:
            obstacle.update(dt)
            
            # FIXED: Check screen position with minimal offset effect
            screen_x = obstacle.world_x - (self.world_offset_x * 0.05)  # Minimal effect
            if obstacle.y < self.screen_height + 100 and -100 <= screen_x <= self.screen_width + 100:
                active_obstacles.append(obstacle)
        
        self.obstacles = active_obstacles
        return active_obstacles

    def spawn_obstacle(self):
        """Spawn a new obstacle at fixed world position"""
        if len(self.obstacles) < 10:  # Limit number of obstacles
            # FIXED: Spawn at fixed world coordinates - no offset compensation
            world_x = random.randint(self.road_x + 50, self.road_x + self.road_width - 50)
            y = -50
            obstacle_type = random.choice(["cone", "rock", "barrier"])
            new_obstacle = Obstacle(world_x, y, obstacle_type, self.obstacle_speed)
            new_obstacle.world_x = world_x  # Store fixed world position
            self.obstacles.append(new_obstacle)
            return True
        return False

    def clear_obstacles(self):
        """Clear all obstacles"""
        self.obstacles = []