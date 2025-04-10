#!/usr/bin/env python
"""
Car Control Module

This module handles the conversion of hand gestures to car control commands.
It provides a clean interface between the hand detection and the car physics.
"""

import math
import numpy as np

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
    
    def update(self, controls, dt):
        """
        Update car position and state based on controls and time delta
        
        Args:
            controls: Dictionary with control commands
            dt: Time delta in seconds
        """
        # Extract controls
        target_direction = float(controls.get('steering', 0.0))
        target_speed = float(controls.get('throttle', 0.0))
        self.braking = bool(controls.get('braking', False))
        self.boost_active = bool(controls.get('boost', False))
        
        # Update direction with smooth transition
        self.direction = self.lerp(self.direction, target_direction, self.steering_speed)
        
        # Update speed based on controls
        if self.braking:
            # Apply brakes (rapid deceleration)
            self.speed = max(0.0, self.speed - self.brake_deceleration * dt)
        elif self.boost_active:
            # Apply boost (accelerate faster to max)
            self.speed = min(1.0, self.speed + self.acceleration * 2 * dt)
        else:
            # Normal acceleration/deceleration
            if target_speed > self.speed:
                self.speed = min(target_speed, self.speed + self.acceleration * dt)
            else:
                self.speed = max(target_speed, self.speed - self.deceleration * dt)
        
        # Calculate actual pixel movement
        speed_pixels_per_second = self.max_speed * self.speed
        if self.boost_active:
            speed_pixels_per_second *= self.boost_multiplier
        
        move_amount = speed_pixels_per_second * dt
        
        # Calculate how much to turn based on speed and direction
        turn_factor = self.direction * move_amount * 0.05
        
        # Update position
        self.x += turn_factor
        self.y -= move_amount  # Moving up = negative y
        
        # Update collision points
        self.update_collision_points()
    
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