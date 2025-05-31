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
    
    def draw(self, screen):
        """Draw the obstacle"""
        color = (255, 0, 0) if self.hit else self.color
        pygame.draw.rect(screen, color, self.rect)

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
    
    def set_road_parameters(self, screen_width, screen_height, road_width, road_x):
        """Set road parameters after initialization"""
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.road_width = road_width
        self.road_x = road_x
    
    def update(self, dt, score):
        """Update all obstacles"""
        active_obstacles = []
        for obstacle in self.obstacles:
            obstacle.update(dt)
            if obstacle.y < self.screen_height + 100:  # Keep if still on screen
                active_obstacles.append(obstacle)
        
        self.obstacles = active_obstacles
        return active_obstacles
    
    def spawn_obstacle(self):
        """Spawn a new obstacle"""
        if len(self.obstacles) < 10:  # Limit number of obstacles
            x = random.randint(self.road_x + 50, self.road_x + self.road_width - 50)
            y = -50
            obstacle_type = random.choice(["cone", "rock", "barrier"])
            new_obstacle = Obstacle(x, y, obstacle_type, self.obstacle_speed)
            self.obstacles.append(new_obstacle)
            return True
        return False
    
    def clear_obstacles(self):
        """Clear all obstacles"""
        self.obstacles = []
