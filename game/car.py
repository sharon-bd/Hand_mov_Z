#!/usr/bin/env python
"""
Car Module for Hand Gesture Car Control Game

This module implements the Car class for the game.
"""

import math
import pygame
import time

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
        
        # Anti-spin steering parameters - more aggressive values
        self.steering_sensitivity = 1.5   # Reduced from 2.5 to 1.5
        self.max_steering_angle = 20      # Reduced from 35 to 20 degrees
        self.steering_return_factor = 0.1 # Increased from 0.05 to 0.1
        self.max_turn_rate = 45           # Maximum degrees per second the car can turn
        self.steering_deadzone = 0.1      # Ignore very small steering inputs
        
        # Anti-spin detection
        self.rotation_history = []        # Track rotation changes to detect spinning
        self.rotation_history_max = 20    # Number of frames to track
        self.spinning_threshold = 540     # Total degrees of rotation before anti-spin kicks in
        self.last_rotation = 0            # Last frame's rotation
        
        # Screen boundaries
        self.screen_width = 800  # Default screen width
        self.screen_height = 600 # Default screen height
        self.boundary_padding = 50 # Keep car this far from edge
        
        # Special states
        self.boost_active = False
        self.braking = False
        
        # Appearance
        self.color = (50, 50, 200)  # Blue car
        self.show_indicators = True
        
        # Physics properties
        self.mass = 1000  # kg
        self.rotation = 0  # degrees
        self.target_rotation = 0  # Target rotation angle based on steering
        
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
            Car._debug_enabled = True  # Enable debug by default
            Car._last_debug_time = time.time()
        
        # Increment counter and only log occasionally
        Car._debug_counter += 1
        current_time = time.time()
        if Car._debug_enabled and current_time - Car._last_debug_time > 2.0:  # Log every 2 seconds
            Car._last_debug_time = current_time
            Car._debug_counter = 0
            # Only print part of the controls to reduce output size
            brief_controls = {
                'steering': controls.get('steering', 0),
                'throttle': controls.get('throttle', 0),
                'braking': controls.get('braking', False),
                'boost': controls.get('boost', False),
                'gesture_name': controls.get('gesture_name', 'Unknown')
            }
            print(f"🚗 Car receiving controls: {brief_controls}")
            print(f"🚗 Current car state: pos=({self.x:.1f},{self.y:.1f}), speed={self.speed:.2f}, rotation={self.rotation:.1f}°")
        
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
            
            # Apply steering deadzone - ignore small values
            if abs(steering) < self.steering_deadzone:
                steering = 0.0
            
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
            
            # ===== ENHANCED: More aggressive anti-spin physics =====
            
            # Store rotation from last frame
            previous_rotation = self.rotation
            
            # Apply steering only if moving (more aggressive speed dependency)
            if self.speed > 0.05:
                # Calculate steering effect with much stronger speed-based reduction
                steering_effect = self.direction * self.steering_sensitivity
                
                # Scale steering effect down dramatically at high speeds
                speed_factor = max(0.2, 1.0 - (self.speed * 1.5))  # More aggressive scaling
                steering_effect *= speed_factor
                
                # Calculate maximum rotation change based on speed and our max_turn_rate
                max_angle_change = self.max_steering_angle * self.speed
                max_rate_limited_change = self.max_turn_rate * dt  # Limit by degrees/second
                
                # Use the smaller of the two limits
                max_allowed_change = min(max_angle_change, max_rate_limited_change)
                
                # Apply the limited steering effect
                rotation_change = min(max_allowed_change, 
                                     max(-max_allowed_change, 
                                         steering_effect * max_allowed_change))
                
                # Apply rotation change
                self.rotation += rotation_change
                
                # ===== NEW: Detect and correct continuous spinning =====
                
                # Track rotation history to detect spinning
                self.rotation_history.append(self.rotation)
                if len(self.rotation_history) > self.rotation_history_max:
                    self.rotation_history.pop(0)
                
                # Calculate total rotation in history
                if len(self.rotation_history) > 5:  # Need at least a few frames
                    # Detect continuous rotation in one direction
                    rot_diff = (self.rotation - self.last_rotation + 180) % 360 - 180
                    
                    # If we've been turning in the same direction for too long, apply correction
                    if abs(rot_diff) > 0.1:  # Non-zero rotation
                        # Strong correction if steering is still at maximum but we've turned a lot
                        if abs(steering) > 0.9 and abs(self.rotation - self.target_rotation) > 90:
                            # Force a strong return to one of 8 cardinal directions
                            target_angle = round(self.rotation / 45) * 45
                            correction = (target_angle - self.rotation) * 0.2  # Strong correction
                            self.rotation += correction * dt * 5  # Apply correction 5x stronger
                
                # Apply natural return to center when steering is not at maximum
                if abs(self.direction) < 0.8:  # Increased threshold for return to center
                    # Find the closest rotation that's a multiple of 45 degrees
                    self.target_rotation = round(self.rotation / 45) * 45
                    # Gradually rotate toward that target - stronger return to center
                    angle_diff = (self.target_rotation - self.rotation) * self.steering_return_factor
                    self.rotation += angle_diff * dt * 2  # Apply 2x stronger return to center
                
                # Store current rotation for next frame
                self.last_rotation = self.rotation
            
            # Keep rotation in 0-360 range
            self.rotation %= 360
            
            # חישוב וקטור תנועה
            rad = math.radians(self.rotation)
            distance = movement_speed * dt
            dx = math.sin(rad) * distance
            dy = -math.cos(rad) * distance
            
            # עדכון מיקום עם בדיקת גבולות
            new_x = self.x + dx
            new_y = self.y + dy
            
            # ===== IMPROVED: Strict boundary checking to keep car on screen =====
            # Get screen dimensions - use instance values or defaults
            screen_width = getattr(self, 'screen_width', 800)
            screen_height = getattr(self, 'screen_height', 600)
            
            # Check if we're about to hit any boundary
            hit_boundary = False
            
            # Check and enforce X boundaries - never allow going beyond the padding
            if new_x < self.boundary_padding:
                new_x = self.boundary_padding
                hit_boundary = True
            elif new_x > screen_width - self.boundary_padding:
                new_x = screen_width - self.boundary_padding
                hit_boundary = True
                
            # Check and enforce Y boundaries - never allow going beyond the padding
            if new_y < self.boundary_padding:
                new_y = self.boundary_padding
                hit_boundary = True
            elif new_y > screen_height - self.boundary_padding:
                new_y = screen_height - self.boundary_padding
                hit_boundary = True
            
            # If we hit any boundary, reduce speed and add visual bounce effect
            if hit_boundary:
                # Stronger speed reduction at boundaries
                self.speed *= 0.4  # More aggressive slowdown
                
                # Create a small "bounce" effect by reversing a tiny bit of velocity
                # This makes the collision feel more realistic
                bounce_factor = -0.2  # Small rebound
                if new_x == self.boundary_padding or new_x == screen_width - self.boundary_padding:
                    # Modify the x-velocity component slightly for horizontal bounce
                    self.x += dx * bounce_factor
                if new_y == self.boundary_padding or new_y == screen_height - self.boundary_padding:
                    # Modify the y-velocity component slightly for vertical bounce
                    self.y += dy * bounce_factor
                
            # Always update position to the clamped values
            self.x = new_x
            self.y = new_y
            
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
    
    def set_screen_dimensions(self, width, height):
        """
        Set the screen dimensions for boundary checking
        
        Args:
            width: Screen width
            height: Screen height
        """
        self.screen_width = width
        self.screen_height = height