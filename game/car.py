#!/usr/bin/env python
"""
Car Module for Hand Gesture Car Control Game

This module implements the Car class for the game.
"""

import math
import pygame

class Car:
    """
    Represents a car in the game
    """
    
    def __init__(self, x, y, width=40, height=80):
        """
        Initialize a car
        
        Args:
            x, y: Initial position
            width, height: Car dimensions
        """
        # Position and size
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        
        # Movement parameters
        self.direction = 0.0  # -1.0 (left) to 1.0 (right)
        self.speed = 0.0      # 0.0 to 1.0
        self.max_speed = 300  # Maximum speed in pixels per second
        self.boost_multiplier = 1.5  # Speed multiplier when boosting
        self.brake_deceleration = 0.4  # ערך האטה כשבולמים
        
        # Special states
        self.boost_active = False
        self.braking = False
        
        # Appearance
        self.color = (50, 50, 200)  # Blue car
        self.show_indicators = True
        
        # Physics properties
        self.mass = 1000  # kg
        self.rotation = 0  # degrees
        
        # Collision detection
        self.collision_points = []
        self.update_collision_points()
        
        # Additional properties
        self.health = 100
        self.score = 0
        
        # History for trail effect
        self.position_history = []
        self.max_history = 20
    
    def update(self, controls, dt):
        """
        עדכון מצב המכונית על סמך הבקרות
        
        Args:
            controls: מילון עם פקודות בקרה
            dt: פרק זמן בשניות
        """
        # Static counter variable for all Car instances
        if not hasattr(Car, '_debug_counter'):
            Car._debug_counter = 0
            Car._debug_enabled = False  # Disabled by default
        
        # Increment counter and only log occasionally
        Car._debug_counter += 1
        if Car._debug_enabled and Car._debug_counter >= 120:  # Even less frequent logging (120 frames)
            Car._debug_counter = 0
            # Only print part of the controls to reduce output size
            brief_controls = {
                'steering': controls.get('steering', 0),
                'throttle': controls.get('throttle', 0)
            }
            print(f"DEBUG - Car receiving brief controls: {brief_controls}")
        
        # חילוץ בקרות עם בדיקת שגיאות נאותה
        try:
            # הערכים צריכים להיות בטווח הנכון כבר בשלב זה - אם לא, נבדוק אותם שוב
            steering = float(controls.get('steering', 0.0))
            throttle = float(controls.get('throttle', 0.0))
            braking = bool(controls.get('braking', False))
            boost = bool(controls.get('boost', False))
            
            # בדיקה נוספת שהערכים בטווח הנכון
            steering = max(-1.0, min(1.0, steering))
            throttle = max(0.0, min(1.0, throttle))
            
            # עדכון המכונית
            self.direction = steering
            
            # עדכון מהירות בהתאם למצערת
            speed_change_rate = 0.1 if throttle > self.speed else 0.2
            self.speed = self.speed + (throttle - self.speed) * speed_change_rate
            
            # טיפול בבלימה
            if braking:
                self.speed = max(0.0, self.speed - self.brake_deceleration * dt)
            
            # חישוב תנועה בפועל
            movement_speed = self.max_speed * self.speed
            if boost:
                movement_speed *= self.boost_multiplier
            
            # עדכון מיקום על סמך כיוון ומהירות
            angle = self.direction * math.pi/4
            rotation_speed = self.speed * 100 * dt
            
            self.rotation += self.direction * rotation_speed
            self.rotation %= 360
            
            # חישוב וקטור תנועה
            rad = math.radians(self.rotation)
            distance = movement_speed * dt
            dx = math.sin(rad) * distance
            dy = -math.cos(rad) * distance
            
            # עדכון מיקום עם בדיקת גבולות
            self.x += dx
            self.y += dy
            
            # עדכון נקודות התנגשות
            self.update_collision_points()
            
            # שמירת מיקום להיסטוריה (לאפקט שובל)
            if distance > 0:
                self.position_history.append((self.x, self.y))
                if len(self.position_history) > self.max_history:
                    self.position_history.pop(0)
        
        except Exception as e:
            print(f"שגיאה בעדכון המכונית: {e}")
            
    def draw(self, screen):
        """
        Draw the car on the screen
        
        Args:
            screen: Pygame surface to draw on
        """
        # Draw trail effect if moving
        if self.position_history and self.speed > 0.1:
            for i in range(len(self.position_history) - 1):
                alpha = i / len(self.position_history) * 255
                color = (50, 50, min(50 + int(alpha), 255))
                width = max(1, int(i / len(self.position_history) * 3))
                
                pygame.draw.line(
                    screen, 
                    color, 
                    self.position_history[i], 
                    self.position_history[i+1], 
                    width
                )
        
        # Create a rotated car surface
        car_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # Choose color based on state
        color = self.color
        if self.boost_active:
            color = (0, 150, 255)  # Bluish boost color
        elif self.braking:
            color = (200, 50, 50)  # Reddish brake color
        
        # Draw car body
        pygame.draw.rect(
            car_surface, 
            color, 
            (0, 0, self.width, self.height),
            0,  # Fill rectangle
            10  # Rounded corners
        )
        
        # Draw windshield
        windshield_width = self.width * 0.7
        windshield_height = self.height * 0.3
        pygame.draw.rect(
            car_surface,
            (150, 220, 255),  # Light blue windshield
            (
                (self.width - windshield_width) / 2,
                self.height * 0.15,
                windshield_width,
                windshield_height
            ),
            0,  # Fill rectangle
            5   # Slightly rounded corners
        )
        
        # Draw headlights
        light_size = self.width // 5
        # Left headlight
        pygame.draw.circle(
            car_surface,
            (255, 255, 200),  # Yellowish light
            (self.width // 4, light_size),
            light_size // 2
        )
        # Right headlight
        pygame.draw.circle(
            car_surface,
            (255, 255, 200),  # Yellowish light
            (self.width - self.width // 4, light_size),
            light_size // 2
        )
        
        # Draw brake lights if braking
        if self.braking:
            # Left brake light
            pygame.draw.circle(
                car_surface,
                (255, 50, 50),  # Red light
                (self.width // 4, self.height - light_size),
                light_size // 2
            )
            # Right brake light
            pygame.draw.circle(
                car_surface,
                (255, 50, 50),  # Red light
                (self.width - self.width // 4, self.height - light_size),
                light_size // 2
            )
        
        # Draw boost effect if active
        if self.boost_active:
            flame_points = [
                (self.width // 2, self.height),
                (self.width // 2 - self.width // 4, self.height + self.height // 3),
                (self.width // 2 + self.width // 4, self.height + self.height // 3)
            ]
            pygame.draw.polygon(car_surface, (255, 165, 0), flame_points)
            
            # Add inner flame
            inner_flame_points = [
                (self.width // 2, self.height),
                (self.width // 2 - self.width // 8, self.height + self.height // 4),
                (self.width // 2 + self.width // 8, self.height + self.height // 4)
            ]
            pygame.draw.polygon(car_surface, (255, 255, 0), inner_flame_points)
        
        # Rotate the car surface
        rotated_car = pygame.transform.rotate(car_surface, -self.rotation)
        
        # Get the rect of the rotated car and position it
        rotated_rect = rotated_car.get_rect(center=(self.x, self.y))
        
        # Draw the rotated car
        screen.blit(rotated_car, rotated_rect)
        
        # Draw debug indicators if enabled
        if self.show_indicators:
            # Direction indicator
            indicator_length = 50
            dx = math.sin(math.radians(self.rotation)) * indicator_length
            dy = -math.cos(math.radians(self.rotation)) * indicator_length
            pygame.draw.line(
                screen,
                (0, 255, 0),
                (self.x, self.y),
                (self.x + dx, self.y + dy),
                2
            )
            
            # Speed indicator
            pygame.draw.rect(
                screen,
                (0, 255, 0) if not self.boost_active else (255, 165, 0),
                (
                    self.x - self.width//2 - 20,
                    self.y - self.height//2,
                    10,
                    self.height * self.speed
                )
            )
    
    def update_collision_points(self):
        """Update collision detection points around the car"""
        # Calculate points based on car rotation
        rad = math.radians(self.rotation)
        sin_rot = math.sin(rad)
        cos_rot = math.cos(rad)
        
        # Define collision points relative to car center
        half_width = self.width // 2
        half_height = self.height // 2
        
        points = [
            (0, -half_height),  # Front
            (-half_width, -half_height),  # Front left
            (half_width, -half_height),  # Front right
            (0, half_height),  # Rear
            (-half_width, half_height),  # Rear left
            (half_width, half_height),  # Rear right
            (0, 0)  # Center
        ]
        
        # Rotate points and add car position
        self.collision_points = []
        for px, py in points:
            # Rotate point
            rx = px * cos_rot - py * sin_rot
            ry = px * sin_rot + py * cos_rot
            
            # Add car position
            self.collision_points.append((self.x + rx, self.y + ry))
    
    def check_collision(self, obstacle):
        """
        Check for collision with an obstacle
        
        Args:
            obstacle: Dictionary with obstacle position and size
            
        Returns:
            Boolean indicating if a collision occurred
        """
        # Create obstacle rectangle
        obstacle_rect = pygame.Rect(
            obstacle['x'] - obstacle['width']//2,
            obstacle['y'] - obstacle['height']//2,
            obstacle['width'],
            obstacle['height']
        )
        
        # Check if any collision point is inside obstacle
        for point in self.collision_points:
            if obstacle_rect.collidepoint(point):
                return True
        
        return False
    
    def take_damage(self, amount):
        """
        Reduce car health by the given amount
        
        Args:
            amount: Amount of damage to inflict
            
        Returns:
            Boolean indicating if the car is destroyed
        """
        self.health -= amount
        self.health = max(0, self.health)
        return self.health <= 0
    
    def repair(self, amount):
        """
        Repair the car by the given amount
        
        Args:
            amount: Amount of health to restore
        """
        self.health += amount
        self.health = min(100, self.health)
    
    def add_score(self, points):
        """
        Add points to the car's score
        
        Args:
            points: Number of points to add
        """
        self.score += points
        
    def clear_trail(self):
        """
        Clear the position history/trail of the car
        
        This can be useful when resetting the game state
        or when teleporting the car to a new position
        """
        self.position_history = []
        
    def reset_state(self):
        """
        Reset the car to default state
        
        Useful when starting a new game or after a crash
        """
        self.speed = 0.0
        self.direction = 0.0
        self.health = 100
        self.boost_active = False
        self.braking = False
        self.clear_trail()