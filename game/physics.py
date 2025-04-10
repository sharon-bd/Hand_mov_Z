#!/usr/bin/env python
"""
Physics Module for Hand Gesture Car Control Game

This module implements the physics for the car movement and collisions.
"""

import math
import numpy as np

class PhysicsEngine:
    """Handles physics calculations for the game"""
    
    def __init__(self):
        """Initialize the physics engine with default settings"""
        # Physics constants
        self.gravity = 9.8  # m/s²
        self.friction = 0.7  # coefficient of friction
        self.drag_coefficient = 0.3  # air resistance
        
        # Simulation settings
        self.time_scale = 1.0  # time scale factor
        self.simulation_steps = 1  # physics steps per frame
    
    def apply_forces(self, car, dt):
        """
        Apply physics forces to a car
        
        Args:
            car: The car object to update
            dt: Time delta in seconds
        """
        # Scale time delta by time scale
        scaled_dt = dt * self.time_scale
        
        # Apply drag (air resistance)
        drag_force = car.speed ** 2 * self.drag_coefficient
        car.speed = max(0, car.speed - drag_force * scaled_dt)
        
        # Apply road friction based on speed
        friction_force = car.speed * self.friction
        car.speed = max(0, car.speed - friction_force * scaled_dt)
    
    def update_car_physics(self, car, controls, dt):
        """
        Update car position and orientation based on physics
        
        Args:
            car: The car object to update
            controls: Dictionary of control inputs
            dt: Time delta in seconds
        """
        # Get smaller time step for more accurate physics
        sub_dt = dt / self.simulation_steps
        
        for _ in range(self.simulation_steps):
            # Apply forces
            self.apply_forces(car, sub_dt)
            
            # Update car state based on controls
            self._update_car_state(car, controls, sub_dt)
        
        # Return the updated car state
        return car
    
    def _update_car_state(self, car, controls, dt):
        """Update car state for a single physics step"""
        # Extract controls
        steering = controls.get('steering', 0.0)
        throttle = controls.get('throttle', 0.0)
        braking = controls.get('braking', False)
        boost = controls.get('boost', False)
        
        # Calculate acceleration
        acceleration = 0.0
        
        if braking:
            # Braking force
            acceleration = -car.speed * 2.0
        else:
            # Normal acceleration based on throttle
            acceleration = (throttle - car.speed * 0.5) * car.max_speed
            
            # Apply boost if active
            if boost:
                acceleration *= car.boost_multiplier
        
        # Update speed based on acceleration
        car.speed += acceleration * dt
        car.speed = max(0.0, min(1.0, car.speed))  # Clamp speed
        
        # Calculate effective steering based on speed
        # Cars are harder to turn at high speeds
        effective_steering = steering
        if car.speed > 0.5:
            # Reduce steering effect at high speeds
            effective_steering *= (1.0 - (car.speed - 0.5) * 0.5)
        
        # Calculate turning radius based on speed and steering
        # Slower speeds allow sharper turns
        turn_factor = effective_steering * car.speed * dt * 2.0
        
        # Update direction with turn factor
        car.direction += turn_factor
        car.direction = max(-1.0, min(1.0, car.direction))  # Clamp direction
        
        # Update position based on speed and direction
        speed_pixels = car.max_speed * car.speed * dt
        
        # Calculate movement vector
        angle = car.direction * math.pi / 4  # Convert direction (-1 to 1) to radians
        dx = math.sin(angle) * speed_pixels
        dy = -math.cos(angle) * speed_pixels  # Negative because y increases downwards
        
        # Update car position
        car.x += dx
        car.y += dy
    
    def check_collision(self, obj1, obj2):
        """
        Check for collision between two objects
        
        Args:
            obj1, obj2: Objects with position and size attributes
            
        Returns:
            Boolean indicating if a collision occurred
        """
        # Simple AABB (Axis-Aligned Bounding Box) collision check
        obj1_left = obj1['x'] - obj1['width'] / 2
        obj1_right = obj1['x'] + obj1['width'] / 2
        obj1_top = obj1['y'] - obj1['height'] / 2
        obj1_bottom = obj1['y'] + obj1['height'] / 2
        
        obj2_left = obj2['x'] - obj2['width'] / 2
        obj2_right = obj2['x'] + obj2['width'] / 2
        obj2_top = obj2['y'] - obj2['height'] / 2
        obj2_bottom = obj2['y'] + obj2['height'] / 2
        
        # Check for overlap in both x and y axes
        return (obj1_right >= obj2_left and
                obj1_left <= obj2_right and
                obj1_bottom >= obj2_top and
                obj1_top <= obj2_bottom)
    
    def resolve_collision(self, obj1, obj2, restitution=0.5):
        """
        Resolve a collision between two objects
        
        Args:
            obj1, obj2: Objects involved in collision
            restitution: Bounciness factor (0 = no bounce, 1 = perfect bounce)
            
        Returns:
            Updated objects after collision response
        """
        # Calculate collision normal (direction from obj1 to obj2)
        dx = obj2['x'] - obj1['x']
        dy = obj2['y'] - obj1['y']
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance > 0:
            # Normalize the collision normal
            nx = dx / distance
            ny = dy / distance
            
            # Calculate relative velocity
            rel_vx = obj2['vx'] - obj1['vx']
            rel_vy = obj2['vy'] - obj1['vy']
            
            # Calculate velocity along the normal
            vel_along_normal = rel_vx * nx + rel_vy * ny
            
            # Only resolve if objects are moving toward each other
            if vel_along_normal < 0:
                # Calculate impulse scalar
                impulse = -(1 + restitution) * vel_along_normal
                impulse /= 1/obj1['mass'] + 1/obj2['mass']
                
                # Apply impulse
                obj1['vx'] -= impulse * nx / obj1['mass']
                obj1['vy'] -= impulse * ny / obj1['mass']
                obj2['vx'] += impulse * nx / obj2['mass']
                obj2['vy'] += impulse * ny / obj2['mass']
        
        return obj1, obj2