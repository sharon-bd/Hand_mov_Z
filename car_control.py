#!/usr/bin/env python
"""
Car Control Module

This module handles the conversion of hand gestures to car control commands.
It provides a clean interface between the hand detection and the car physics.
"""

import math
import numpy as np
import pygame  # Add pygame import for collision detection
import time  # Import time for managing hand detection timing

class CarControlHandler:
    """
    Handles the translation of hand gesture data to car control commands
    """
    
    def __init__(self):
        """Initialize the car control handler with default settings"""
        # Control smoothing parameters
        self.steering_smoothing = 0.5  # Higher value = smoother but less responsive
        self.throttle_smoothing = 0.3  # Higher value = smoother but less responsive
        
        # Previous control values for smoothing
        self.prev_steering = 0.0
        self.prev_throttle = 0.0
        
        # Control sensitivity
        self.steering_sensitivity = 1.0
        self.throttle_sensitivity = 1.0
        
        # Control limits
        self.max_steering_rate = 0.1  # Maximum steering change per update
        
        # Minimum and maximum speed settings
        self.min_speed = 0.2  # Minimum speed (20%)
        self.max_speed = 1.0  # Maximum speed (100%)
        
        # No hand detected timeout
        self.no_hand_timeout = 2.0  # Seconds before dropping to minimum speed
        self.last_hand_time = time.time()
        
        # Current throttle and steering values
        self.current_throttle = self.min_speed
        self.current_steering = 0.0
        
    def process_controls(self, gesture_data):
        """
        Process raw gesture data into smooth car control commands
        
        Args:
            gesture_data: Dictionary containing raw control data
                          ('steering', 'throttle', 'braking', 'boost')
        
        Returns:
            Dictionary with processed control commands
        """
        # Extract raw control values
        raw_steering = float(gesture_data.get('steering', 0.0))
        raw_throttle = float(gesture_data.get('throttle', 0.0))
        braking = bool(gesture_data.get('braking', False))
        boost = bool(gesture_data.get('boost', False))
        
        # Apply sensitivity
        adjusted_steering = raw_steering * self.steering_sensitivity
        adjusted_throttle = raw_throttle * self.throttle_sensitivity
        
        # Limit rate of steering change to prevent jerky movement
        steering_change = adjusted_steering - self.prev_steering
        if abs(steering_change) > self.max_steering_rate:
            steering_change = math.copysign(self.max_steering_rate, steering_change)
        
        limited_steering = self.prev_steering + steering_change
        
        # Apply smoothing
        smooth_steering = self.apply_smoothing(self.prev_steering, limited_steering, self.steering_smoothing)
        smooth_throttle = self.apply_smoothing(self.prev_throttle, adjusted_throttle, self.throttle_smoothing)
        
        # Update previous values
        self.prev_steering = smooth_steering
        self.prev_throttle = smooth_throttle
        
        # Create processed controls
        controls = {
            'steering': smooth_steering,
            'throttle': smooth_throttle,
            'braking': braking,
            'boost': boost,
            'gear': 'D'  # Default to Drive
        }
        
        # Special handling for braking - override throttle
        if braking:
            controls['throttle'] = 0.0
            controls['gear'] = 'B'  # Brake
        
        return controls
    
    def apply_smoothing(self, previous, current, smoothing_factor):
        """
        Apply smoothing between previous and current value
        
        Args:
            previous: Previous value
            current: Current value
            smoothing_factor: How much to weight the previous value (0-1)
        
        Returns:
            Smoothed value
        """
        return previous * smoothing_factor + current * (1.0 - smoothing_factor)
    
    def reset(self):
        """Reset all control states"""
        self.prev_steering = 0.0
        self.prev_throttle = 0.0


class Car:
    """
    Represents a car in the game with physics-based movement
    """
    
    def __init__(self, x, y, width=40, height=80):
        """
        Initialize the car
        
        Args:
            x, y: Initial position
            width, height: Car dimensions
        """
        # Position and dimensions
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        
        # Movement properties
        self.direction = 0.0  # -1.0 (left) to 1.0 (right)
        self.speed = 0.0      # 0.0 to 1.0
        self.max_speed = 300  # Maximum speed in pixels per second
        self.acceleration = 0.1  # Acceleration rate
        self.deceleration = 0.2  # Deceleration rate
        self.steering_speed = 0.15  # How quickly steering changes take effect
        
        # Special states
        self.boost_active = False
        self.boost_multiplier = 1.5
        self.braking = False
        self.brake_deceleration = 0.4  # Faster deceleration when braking
        
        # Visual properties
        self.color = (50, 50, 200)  # Blue car
        self.show_indicators = True
        
        # Collision detection
        self.collision_points = []
        self.update_collision_points()
        
        # Rotation and position history for visual effects
        self.rotation = 0  # Rotation in degrees
        self.position_history = []
        self.max_history = 10  # Store last 10 positions for trail effect
        
        # Add debug info
        self.debug_info = {
            'last_controls': {},
            'normalized_controls': {},
            'boundary_hit': False
        }
    
    def update(self, controls, dt):
        """
        Update car state based on controls
        
        Args:
            controls: Dictionary with control commands
            dt: Time delta in seconds
        """
        try:
            # Store original controls for debugging
            self.debug_info['last_controls'] = controls.copy()
            
            # Fix debug print statements format to avoid potential errors
            print(f"DEBUG - Car receiving: steering={controls.get('steering', 0.0)}, "
                  f"throttle={controls.get('throttle', 0.0)}")
            
            # Add a boundary check to keep car on screen
            screen_width = 800  # Default screen width
            screen_height = 600  # Default screen height
            
            # Extract controls with proper error checking and normalization
            try:
                steering_input = float(controls.get('steering', 0.0))
                throttle_input = float(controls.get('throttle', 0.0))
                self.braking = bool(controls.get('braking', False))
                self.boost_active = bool(controls.get('boost', False))
                
                # Ensure values are in the right range - add explicit clamping
                steering_input = max(-1.0, min(1.0, steering_input))
                throttle_input = max(0.0, min(1.0, throttle_input))
                
                # Set the direction property directly
                self.direction = steering_input
                
                # Store normalized controls for debugging
                self.debug_info['normalized_controls'] = {
                    'steering': self.direction,
                    'throttle': throttle_input,
                    'braking': self.braking,
                    'boost': self.boost_active
                }
            except (ValueError, TypeError) as e:
                print(f"ERROR: Invalid control values: {e}")
                # Use safe defaults if conversion fails
                self.direction = 0.0
                throttle_input = 0.0
                self.braking = False
                self.boost_active = False
            
            # Apply smoothing to speed changes - more responsive rates
            speed_change_rate = 0.2 if throttle_input > self.speed else 0.3
            self.speed = self.speed + (throttle_input - self.speed) * speed_change_rate
            
            # Special case for braking
            if self.braking:
                self.speed = max(0.0, self.speed - self.brake_deceleration * dt)
            
            # Calculate actual movement
            movement_speed = self.max_speed * self.speed
            if self.boost_active:
                movement_speed *= self.boost_multiplier
            
            # Update position based on direction and speed
            angle = self.direction * math.pi/4
            rotation_speed = self.speed * 100 * dt
            
            self.rotation += self.direction * rotation_speed
            self.rotation %= 360  # Keep within 0-360
            
            # Convert rotation to radians for movement calculation
            rad = math.radians(self.rotation)
            
            # Calculate movement vector
            distance = movement_speed * dt
            dx = math.sin(rad) * distance
            dy = -math.cos(rad) * distance  # Negative because y increases downwards
            
            # Update position with boundary checking
            new_x = self.x + dx
            new_y = self.y + dy
            
            # Keep car within screen boundaries
            self.debug_info['boundary_hit'] = False
            if new_x < self.width//2 or new_x > screen_width - self.width//2:
                self.debug_info['boundary_hit'] = True
            if new_y < self.height//2 or new_y > screen_height - self.height//2:
                self.debug_info['boundary_hit'] = True
                
            self.x = max(self.width//2, min(screen_width - self.width//2, new_x))
            self.y = max(self.height//2, min(screen_height - self.height//2, new_y))
            
            # Update collision points
            self.update_collision_points()
            
            # Store position history for trail effect
            if distance > 0:  # Only add point if moving
                self.position_history.append((self.x, self.y))
                if len(self.position_history) > self.max_history:
                    self.position_history.pop(0)
        except Exception as e:
            print(f"ERROR in Car.update: {e}")
            import traceback
            traceback.print_exc()
    
    def draw(self, screen):
        """
        Draw the car on the screen
        
        Args:
            screen: Pygame surface to draw on
        """
        # Draw trail/history points if enabled
        if len(self.position_history) > 1:
            for i in range(1, len(self.position_history)):
                # Draw trail with fading opacity
                alpha = int(255 * (i / len(self.position_history)))
                color = (50, 50, 200, alpha)  # Blue trail with alpha
                pygame.draw.line(
                    screen, 
                    color, 
                    self.position_history[i-1], 
                    self.position_history[i], 
                    2
                )
        
        # Save current position and rotation
        save_position = (self.x, self.y)
        save_rotation = self.rotation
        
        # Create a rotated car surface
        car_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        car_surface.fill((0, 0, 0, 0))  # Transparent background
        
        # Draw car body on the surface
        pygame.draw.rect(
            car_surface, 
            self.color, 
            (0, 0, self.width, self.height), 
            0, 
            10  # Rounded corners
        )
        
        # Add details (windshield, lights) if size permits
        if self.width > 20 and self.height > 40:
            # Windshield
            pygame.draw.rect(
                car_surface,
                (150, 220, 255),  # Light blue
                (self.width//6, self.height//6, self.width*2//3, self.height//3),
                0,
                5  # Rounded corners
            )
            
            # Headlights
            light_size = min(8, self.width//5)
            # Left headlight
            pygame.draw.circle(
                car_surface,
                (255, 255, 200),  # Yellowish light
                (self.width//4, self.height//6),
                light_size
            )
            # Right headlight
            pygame.draw.circle(
                car_surface,
                (255, 255, 200),  # Yellowish light
                (self.width*3//4, self.height//6),
                light_size
            )
            
            # Brake lights if braking
            if self.braking:
                # Left brake light
                pygame.draw.circle(
                    car_surface,
                    (255, 50, 50),  # Red
                    (self.width//4, self.height*5//6),
                    light_size
                )
                # Right brake light
                pygame.draw.circle(
                    car_surface,
                    (255, 50, 50),  # Red
                    (self.width*3//4, self.height*5//6),
                    light_size
                )
        
        # Rotate the car surface
        rotated_car = pygame.transform.rotate(car_surface, -self.rotation)
        
        # Get the rect of the rotated surface and center it at car's position
        car_rect = rotated_car.get_rect(center=(self.x, self.y))
        
        # Draw the rotated car
        screen.blit(rotated_car, car_rect)
        
        # Draw boost effect if active
        if self.boost_active and self.speed > 0.1:
            # Calculate flame position at the back of the car
            flame_angle = math.radians((self.rotation + 180) % 360)
            flame_x = self.x + math.sin(flame_angle) * self.height/2
            flame_y = self.y - math.cos(flame_angle) * self.height/2
            
            # Draw flame
            flame_points = [
                (flame_x, flame_y),
                (flame_x + math.sin(flame_angle + 0.3) * 20, flame_y - math.cos(flame_angle + 0.3) * 20),
                (flame_x + math.sin(flame_angle) * 30, flame_y - math.cos(flame_angle) * 30),
                (flame_x + math.sin(flame_angle - 0.3) * 20, flame_y - math.cos(flame_angle - 0.3) * 20)
            ]
            pygame.draw.polygon(
                screen,
                (255, 165, 0),  # Orange flame
                flame_points
            )
        
        # Draw direction indicator if enabled
        if self.show_indicators:
            # Direction indicator from center of car
            indicator_len = max(30, self.width)
            ind_end_x = self.x + math.sin(math.radians(self.rotation)) * indicator_len
            ind_end_y = self.y - math.cos(math.radians(self.rotation)) * indicator_len
            
            pygame.draw.line(
                screen,
                (0, 255, 0),  # Green
                (self.x, self.y),
                (ind_end_x, ind_end_y),
                2
            )
    
    def lerp(self, current, target, factor):
        """Linear interpolation for smoother movement"""
        return current + (target - current) * factor
    
    def update_collision_points(self):
        """Update collision detection points around the car"""
        # Create points around the car for more accurate collision detection
        self.collision_points = [
            # Front center
            (self.x, self.y - self.height//2),
            # Front left corner
            (self.x - self.width//2, self.y - self.height//2),
            # Front right corner
            (self.x + self.width//2, self.y - self.height//2),
            # Rear center
            (self.x, self.y + self.height//2),
            # Center point
            (self.x, self.y)
        ]
    
    def check_collision(self, obstacle):
        """
        Check if the car collides with the given obstacle
        
        Args:
            obstacle: Dictionary with obstacle properties (x, y, width, height)
        
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
        
        # Check if any of the collision points are within the obstacle
        for point in self.collision_points:
            if obstacle_rect.collidepoint(point):
                return True
        
        # Also check car rectangle for complete coverage
        car_rect = pygame.Rect(
            self.x - self.width//2,
            self.y - self.height//2,
            self.width,
            self.height
        )
        
        return car_rect.colliderect(obstacle_rect)
    
    def update_car_controls(self, gesture_data):
        """עדכון בקרת המכונית - יותר יציב"""
        current_time = time.time()
        
        # אתחול ברירת מחדל
        if not hasattr(self, 'current_throttle'):
            self.current_throttle = 0.3  # מהירות התחלתית בטוחה
            self.current_steering = 0
            self.last_hand_time = current_time
        
        if gesture_data and gesture_data.get('hand_detected', False):
            self.last_hand_time = current_time
            
            # קבל ערכי בקרה עם ברירת מחדל בטוחה
            throttle = gesture_data.get('throttle', 0.3)
            steering = gesture_data.get('steering', 0)
            
            # החלק שינויים פתאומיים
            throttle_diff = throttle - self.current_throttle
            steering_diff = steering - self.current_steering
            
            # הגבל שינויים פתאומיים
            max_throttle_change = 0.1
            max_steering_change = 0.2
            
            if abs(throttle_diff) > max_throttle_change:
                throttle = self.current_throttle + (max_throttle_change if throttle_diff > 0 else -max_throttle_change)
            
            if abs(steering_diff) > max_steering_change:
                steering = self.current_steering + (max_steering_change if steering_diff > 0 else -max_steering_change)
            
            self.current_throttle = max(0.2, min(1.0, throttle))  # מינימום 0.2
            self.current_steering = max(-1.0, min(1.0, steering))
            
        else:
            # אין יד - שמור על מהירות מינימלית
            time_without_hand = current_time - self.last_hand_time
            
            if time_without_hand > 1.0:  # אחרי שנייה
                # ירידה הדרגתית למהירות מינימלית
                target_throttle = 0.3
                if self.current_throttle > target_throttle:
                    self.current_throttle = max(target_throttle, self.current_throttle - 0.02)
                
                # איפוס הגה הדרגתי
                if abs(self.current_steering) > 0.01:
                    self.current_steering *= 0.9
    
        return {
            'throttle': self.current_throttle,
            'steering': self.current_steering,
            'brake': False
        }