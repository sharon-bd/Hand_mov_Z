#!/usr/bin/env python
"""
Game Objects Module for Hand Gesture Car Control Game

This module implements various game objects like obstacles, power-ups, etc.
"""

import random
import math
import pygame

class ObstacleManager:
    """Manages the creation and updating of obstacles in the game"""
    
    def __init__(self, screen_width, screen_height, road_width, road_x):
        """
        Initialize the obstacle manager
        
        Args:
            screen_width: Width of the game screen
            screen_height: Height of the game screen
            road_width: Width of the road
            road_x: X position of the left edge of the road
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.road_width = road_width
        self.road_x = road_x
        
        # List of active obstacles
        self.obstacles = []
        
        # Obstacle types
        self.obstacle_types = [
            {
                "name": "cone",
                "width": 20,
                "height": 30,
                "color": (255, 140, 0),
                "points": 10,
                "damage": 5
            },
            {
                "name": "barrier",
                "width": 80,
                "height": 15,
                "color": (255, 50, 50),
                "points": 20,
                "damage": 15
            },
            {
                "name": "rock",
                "width": 40,
                "height": 40,
                "color": (120, 120, 120),
                "points": 15,
                "damage": 10
            }
        ]
        
        # Spawn settings
        self.spawn_rate = 0.02  # Probability of spawning per frame
        self.min_obstacle_distance = 100  # Minimum distance between obstacles
        self.obstacle_speed = 200  # Speed in pixels per second
        self.max_obstacles = 15  # Maximum number of obstacles at once
    
    def update(self, dt, score):
        """
        Update all obstacles
        
        Args:
            dt: Time delta in seconds
            score: Current score (affects difficulty)
            
        Returns:
            List of obstacles that are still active
        """
        # Move obstacles down the screen
        # The higher the score, the faster they move
        speed_multiplier = 1.0 + score / 1000  # Increase speed by 10% per 100 points
        speed = self.obstacle_speed * speed_multiplier
        
        active_obstacles = []
        for obstacle in self.obstacles:
            # Move obstacle down
            obstacle['y'] += speed * dt
            
            # Keep if still on screen
            if obstacle['y'] < self.screen_height + obstacle['height']:
                active_obstacles.append(obstacle)
        
        # Update obstacle list
        self.obstacles = active_obstacles
        
        return active_obstacles
    
    def spawn_obstacle(self):
        """
        Spawn a new obstacle if conditions are met
        
        Returns:
            True if an obstacle was spawned
        """
        # Don't spawn if we've reached the maximum
        if len(self.obstacles) >= self.max_obstacles:
            return False
        
        # Random chance to spawn
        if random.random() > self.spawn_rate:
            return False
        
        # Select a random obstacle type
        obstacle_type = random.choice(self.obstacle_types)
        
        # Select a random lane
        road_lanes = 3
        lane_width = self.road_width / road_lanes
        lane = random.randint(0, road_lanes - 1)
        
        # Calculate x position (center of lane)
        x = self.road_x + (lane + 0.5) * lane_width
        
        # Add some random variation to position
        x += random.uniform(-lane_width * 0.2, lane_width * 0.2)
        
        # Check if there's enough space from other obstacles
        for obstacle in self.obstacles:
            dist = math.sqrt((obstacle['x'] - x)**2 + (obstacle['y'] - 0)**2)
            if dist < self.min_obstacle_distance:
                return False
        
        # Create the obstacle
        new_obstacle = {
            'x': x,
            'y': -obstacle_type['height'],  # Start just above the screen
            'width': obstacle_type['width'],
            'height': obstacle_type['height'],
            'type': obstacle_type['name'],
            'color': obstacle_type['color'],
            'points': obstacle_type['points'],
            'damage': obstacle_type['damage'],
            'hit': False
        }
        
        # Add to obstacle list
        self.obstacles.append(new_obstacle)
        
        return True
    
    def clear_obstacles(self):
        """Clear all obstacles"""
        self.obstacles = []
    
    def draw_obstacles(self, screen):
        """
        Draw all obstacles on the screen
        
        Args:
            screen: Pygame surface to draw on
        """
        for obstacle in self.obstacles:
            self._draw_single_obstacle(screen, obstacle)
    
    def _draw_single_obstacle(self, screen, obstacle):
        """Draw a single obstacle based on its type"""
        # Choose color - highlight if hit
        color = (255, 0, 0) if obstacle.get('hit', False) else obstacle['color']
        
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
                (255, 255, 255),
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
                stripe_color = (255, 255, 255) if i % 2 == 0 else (255, 0, 0)
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
                (color[0] - 30, color[1] - 30, color[2] - 30),
                (int(obstacle['x'] + 5), int(obstacle['y'] - 5)),
                obstacle['width'] // 4
            )


class PowerUpManager:
    """Manages the creation and updating of power-ups in the game"""
    
    def __init__(self, screen_width, screen_height, road_width, road_x):
        """
        Initialize the power-up manager
        
        Args:
            screen_width: Width of the game screen
            screen_height: Height of the game screen
            road_width: Width of the road
            road_x: X position of the left edge of the road
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.road_width = road_width
        self.road_x = road_x
        
        # List of active power-ups
        self.power_ups = []
        
        # Power-up types
        self.power_up_types = [
            {
                "name": "boost",
                "width": 30,
                "height": 30,
                "color": (255, 215, 0),  # Gold
                "duration": 5.0,  # seconds
                "effect": "boost"
            },
            {
                "name": "shield",
                "width": 30,
                "height": 30,
                "color": (0, 191, 255),  # Deep sky blue
                "duration": 8.0,
                "effect": "shield"
            },
            {
                "name": "repair",
                "width": 30,
                "height": 30,
                "color": (0, 255, 0),  # Green
                "duration": 0.0,  # Instant effect
                "effect": "repair"
            }
        ]
        
        # Spawn settings
        self.spawn_rate = 0.005  # Probability of spawning per frame
        self.min_powerup_distance = 150
        self.powerup_speed = 150
        self.max_power_ups = 3
        
        # Animation settings
        self.animation_time = 0
    
    def update(self, dt):
        """
        Update all power-ups
        
        Args:
            dt: Time delta in seconds
            
        Returns:
            List of power-ups that are still active
        """
        # Update animation time
        self.animation_time += dt
        
        # Move power-ups down the screen
        active_power_ups = []
        for power_up in self.power_ups:
            # Move power-up down
            power_up['y'] += self.powerup_speed * dt
            
            # Update animation
            power_up['animation_offset'] = math.sin(self.animation_time * 5) * 5
            
            # Keep if still on screen
            if power_up['y'] < self.screen_height + power_up['height']:
                active_power_ups.append(power_up)
        
        # Update power-up list
        self.power_ups = active_power_ups
        
        return active_power_ups
    
    def spawn_power_up(self):
        """
        Spawn a new power-up if conditions are met
        
        Returns:
            True if a power-up was spawned
        """
        # Don't spawn if we've reached the maximum
        if len(self.power_ups) >= self.max_power_ups:
            return False
        
        # Random chance to spawn
        if random.random() > self.spawn_rate:
            return False
        
        # Select a random power-up type
        power_up_type = random.choice(self.power_up_types)
        
        # Select a random position on the road
        x = random.uniform(self.road_x + 50, self.road_x + self.road_width - 50)
        
        # Check if there's enough space from other objects
        for power_up in self.power_ups:
            dist = math.sqrt((power_up['x'] - x)**2 + (power_up['y'] - 0)**2)
            if dist < self.min_powerup_distance:
                return False
        
        # Create the power-up
        new_power_up = {
            'x': x,
            'y': -power_up_type['height'],  # Start just above the screen
            'width': power_up_type['width'],
            'height': power_up_type['height'],
            'type': power_up_type['name'],
            'color': power_up_type['color'],
            'duration': power_up_type['duration'],
            'effect': power_up_type['effect'],
            'animation_offset': 0,
            'collected': False
        }
        
        # Add to power-up list
        self.power_ups.append(new_power_up)
        
        return True
    
    def draw_power_ups(self, screen):
        """
        Draw all power-ups on the screen
        
        Args:
            screen: Pygame surface to draw on
        """
        for power_up in self.power_ups:
            self._draw_single_power_up(screen, power_up)
    
    def _draw_single_power_up(self, screen, power_up):
        """Draw a single power-up with animation"""
        # Get position with animation offset
        x = power_up['x']
        y = power_up['y'] + power_up['animation_offset']
        
        # Base shape is a circle
        pygame.draw.circle(
            screen,
            power_up['color'],
            (int(x), int(y)),
            power_up['width'] // 2
        )
        
        # Add inner circle for glow effect
        pygame.draw.circle(
            screen,
            (255, 255, 255),
            (int(x), int(y)),
            power_up['width'] // 4
        )
        
        # Add icon based on power-up type
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
            pygame.draw.polygon(screen, (0, 0, 0), points)
            
        elif power_up['type'] == 'shield':
            # Draw shield
            pygame.draw.arc(
                screen,
                (0, 0, 0),
                (x - 8, y - 8, 16, 16),
                math.pi / 4,
                math.pi * 7 / 4,
                2
            )
            
        elif power_up['type'] == 'repair':
            # Draw plus sign
            pygame.draw.line(screen, (0, 0, 0), (x - 6, y), (x + 6, y), 2)
            pygame.draw.line(screen, (0, 0, 0), (x, y - 6), (x, y + 6), 2)

            
class Obstacle:
    """Simple obstacle class for the game"""
    
    def __init__(self, x, y):
        """Initialize an obstacle at position x, y"""
        self.x = x
        self.y = y
        self.width = 30
        self.height = 30
        self.color = (255, 0, 0)  # Red by default
        
    def update(self):
        """Update obstacle position"""
        # Basic movement logic - can be enhanced later
        self.y += 2  # Move down
        
    def draw(self, screen):
        """Draw the obstacle"""
        pygame.draw.rect(
            screen, 
            self.color, 
            (self.x - self.width // 2, self.y - self.height // 2, self.width, self.height)
        )


class ScoreManager:
    """Manages scoring and score display"""
    
    def __init__(self):
        """Initialize the score manager"""
        self.score = 0
        self.high_score = 0
        self.score_multiplier = 1.0
        self.obstacles = []  # List to store obstacles
        self.car = None  # Reference to the car will be set later
        
        # Load high score if available
        self._load_high_score()
    
    def add_score(self, points):
        """
        Add points to the score
        
        Args:
            points: Base points to add (will be multiplied by current multiplier)
        """
        self.score += int(points * self.score_multiplier)
        
        # Update high score if needed
        if self.score > self.high_score:
            self.high_score = self.score
            self._save_high_score()
    
    def set_multiplier(self, multiplier):
        """Set the score multiplier"""
        self.score_multiplier = multiplier
    
    def reset_score(self):
        """Reset the current score"""
        self.score = 0
        self.score_multiplier = 1.0
    
    def draw_score(self, screen, font, position=(20, 20), color=(255, 255, 255)):
        """
        Draw the score on the screen
        
        Args:
            screen: Pygame surface to draw on
            font: Pygame font to use
            position: Position to draw the score
            color: Color of the score text
        """
        # Draw current score
        score_text = font.render(f"Score: {self.score}", True, color)
        screen.blit(score_text, position)
        
        # Draw multiplier if not 1.0
        if self.score_multiplier != 1.0:
            mult_text = font.render(f"x{self.score_multiplier:.1f}", True, (255, 215, 0))
            screen.blit(mult_text, (position[0] + score_text.get_width() + 10, position[1]))
        
        # Draw high score
        high_score_text = font.render(f"High Score: {self.high_score}", True, color)
        screen.blit(high_score_text, (position[0], position[1] + 30))
    
    def _load_high_score(self):
        """Load high score from file"""
        try:
            with open("high_score.txt", "r") as f:
                self.high_score = int(f.read().strip())
        except (FileNotFoundError, ValueError):
            self.high_score = 0
    
    def _save_high_score(self):
        """Save high score to file"""
        try:
            with open("high_score.txt", "w") as f:
                f.write(str(self.high_score))
        except:
            pass  # Silently fail if we can't save

    def update_obstacles(self):
        """Update obstacles with simplified logic"""
        
        # Add obstacles at lower frequency
        if random.random() < 0.01:  # Instead of 0.02 - fewer obstacles
            self.add_obstacle()
        
        # Update obstacle positions
        for obstacle in self.obstacles[:]:  # Copy list to prevent errors
            obstacle.update()

            # Remove obstacles that left the screen
            if (obstacle.x < -100 or obstacle.x > self.screen_width + 100 or
                obstacle.y < -100 or obstacle.y > self.screen_height + 100):
                self.obstacles.remove(obstacle)

    def add_obstacle(self):
        """Add new obstacle - far from car"""
        
        # Ensure obstacle is not created too close to car
        min_distance = 150  # Minimum distance from car
        
        attempts = 0
        while attempts < 10:  # Maximum 10 attempts
            x = random.randint(50, self.screen_width - 50)
            y = random.randint(50, self.screen_height - 50)
            
            # Check distance from car
            distance_from_car = math.sqrt((x - self.car.x)**2 + (y - self.car.y)**2)
            
            if distance_from_car > min_distance:
                obstacle = Obstacle(x, y)
                self.obstacles.append(obstacle)
                break
            
            attempts += 1