#!/usr/bin/env python
"""
Car Module for Hand Gesture Car Control Game

This module implements the Car class for the game.
"""

import math
import pygame
import time
import random

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
        self.x = x  # אמיתי בעולם
        self.y = y  # אמיתי בעולם
        self.screen_x = x  # תמיד יהיה במרכז המסך
        self.screen_y = y  # תמיד יהיה במרכז המסך
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
        
        # World boundaries
        self.world_width = 2000  # רוחב העולם הווירטואלי
        self.world_height = 2000  # גובה העולם הווירטואלי
        self.screen_width = 800  # רוחב המסך
        self.screen_height = 600  # גובה המסך
        
        # Collision state
        self.collision_cooldown = 0
        self.collision_flash = False
        self.last_collision_time = 0
        self.is_colliding = False
        
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
                self.braking = True  # עדכון מצב הבלימה של הרכב
            else:
                # הצג אורות בלימה גם כשהמכונית יוצאת ממצב תנועה לעצירה
                self.braking = (self.speed < 0.05 and throttle < 0.1)
            
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
            
            # עדכון מיקום אמיתי בעולם (לא על המסך)
            new_x = self.x + dx
            new_y = self.y + dy
            
            # בדיקת גבולות העולם הווירטואלי
            hit_boundary = False
            
            # בדיקת גבולות אופקיים
            if new_x < 0:
                new_x = 0
                hit_boundary = True
            elif new_x > self.world_width:
                new_x = self.world_width
                hit_boundary = True
                
            # בדיקת גבולות אנכיים
            if new_y < 0:
                new_y = 0
                hit_boundary = True
            elif new_y > self.world_height:
                new_y = self.world_height
                hit_boundary = True
            
            # אם פגענו בגבול, האט את המכונית
            if hit_boundary:
                self.speed *= 0.4  # האטה משמעותית בפגיעה בגבול
                
                # אפקט קפיצה קטן מהקיר
                bounce_factor = -0.2
                if new_x == 0 or new_x == self.world_width:
                    self.x += dx * bounce_factor
                if new_y == 0 or new_y == self.world_height:
                    self.y += dy * bounce_factor
            
            # עדכון המיקום בעולם
            self.x = new_x
            self.y = new_y
            
            # עדכון מצב התנגשות
            if self.collision_cooldown > 0:
                self.collision_cooldown -= dt
                # הבהוב בזמן התנגשות
                self.collision_flash = int(self.collision_cooldown * 10) % 2 == 0
            else:
                self.collision_flash = False
                self.is_colliding = False
            
            # במשחק העדכני המכונית תישאר במרכז המסך
            screen_center_x = self.screen_width // 2
            screen_center_y = self.screen_height - 100 
            self.screen_x = screen_center_x
            self.screen_y = screen_center_y
            
            # עדכון נקודות התנגשות
            self.update_collision_points()
            
            # שמירת מיקום להיסטוריה (לאפקט שובל)
            if distance > 0:
                self.position_history.append((self.x, self.y))  # שמירת מיקום אמיתי בעולם
                if len(self.position_history) > self.max_history:
                    self.position_history.pop(0)
        
        except Exception as e:
            print(f"שגיאה בעדכון המכונית: {e}")
            import traceback
            traceback.print_exc()
            
    def draw(self, screen, offset_x=0, offset_y=0):
        """
        Draw the car on the screen
        
        Args:
            screen: Pygame surface to draw on
            offset_x, offset_y: World offset (for scrolling)
        """
        try:
            # בגרסה זו המכונית תמיד במרכז, ולכן נחשב את המיקום שלה על המסך
            screen_center_x = self.screen_width // 2
            screen_center_y = self.screen_height // 2
            
            # Draw trail effect if moving
            if self.position_history and self.speed > 0.1:
                # Draw trail
                for i, (pos_x, pos_y) in enumerate(self.position_history):
                    # Calculate screen position based on world position
                    screen_x = screen_center_x - (self.x - pos_x)
                    screen_y = screen_center_y - (self.y - pos_y)
                    # Make trail more transparent as it gets older
                    alpha = int(255 * (i / len(self.position_history)))
                    trail_size = max(3, int(self.width * 0.3 * (i / len(self.position_history))))
                    pygame.draw.circle(
                        screen,
                        (*self.color[:3], alpha),  # Use car color with calculated alpha
                        (int(screen_x), int(screen_y)),
                        trail_size
                    )
            
            # Create a rotated car surface
            car_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            
            # Choose color based on state
            color = self.color
            if self.collision_flash:
                # צבע אדום בזמן התנגשות עם מכשול
                color = (255, 0, 0)
            elif self.boost_active:
                color = (0, 150, 255)  # Bluish boost color
            elif self.braking:
                color = (200, 50, 50)  # Reddish brake color
            
            # Draw car body - החזרת הצבע המקורי
            pygame.draw.rect(
                car_surface, 
                color,  # שימוש בצבע המקורי במקום צבע מגנטה לבדיקה
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
            
            # Draw brake lights if braking or stopped
            if self.braking:
                # Left brake light - גדול יותר ובולט יותר
                pygame.draw.circle(
                    car_surface,
                    (255, 30, 30),  # אדום עמוק יותר
                    (self.width // 4, self.height - light_size),
                    light_size // 2 + 2  # גדול יותר ב-2 פיקסלים
                )
                # אפקט זוהר מסביב לאור השמאלי
                pygame.draw.circle(
                    car_surface,
                    (255, 100, 100, 150),  # אדום שקוף יותר
                    (self.width // 4, self.height - light_size),
                    light_size // 2 + 5  # גדול יותר לאפקט זוהר
                )
                
                # Right brake light - גדול יותר ובולט יותר
                pygame.draw.circle(
                    car_surface,
                    (255, 30, 30),  # אדום עמוק יותר
                    (self.width - self.width // 4, self.height - light_size),
                    light_size // 2 + 2  # גדול יותר ב-2 פיקסלים
                )
                # אפקט זוהר מסביב לאור הימני
                pygame.draw.circle(
                    car_surface,
                    (255, 100, 100, 150),  # אדום שקוף יותר
                    (self.width - self.width // 4, self.height - light_size),
                    light_size // 2 + 5  # גדול יותר לאפקט זוהר
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
            rotated_rect = rotated_car.get_rect(center=(screen_center_x, screen_center_y))
            
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
                    (screen_center_x, screen_center_y),
                    (screen_center_x + dx, screen_center_y + dy),
                    2
                )
                
                # Speed indicator
                pygame.draw.rect(
                    screen,
                    (0, 255, 0) if not self.boost_active else (255, 165, 0),
                    (
                        screen_center_x - self.width//2 - 20,
                        screen_center_y - self.height//2,
                        10,
                        self.height * self.speed
                    )
                )
        except Exception as e:
            print(f"❌ Error drawing car: {e}")
            import traceback
            traceback.print_exc()
    
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
    
    def handle_obstacle_collision(self, obstacle_type=None):
        """
        מטפל בהתנגשות עם מכשול
        
        Args:
            obstacle_type: סוג המכשול (אופציונלי)
        """
        # אם יש כבר התנגשות פעילה, לא נתייחס לזו החדשה
        if self.collision_cooldown > 0:
            return
        
        # קביעת משך ההתנגשות והשפעתה לפי סוג המכשול
        if obstacle_type == "rock":
            self.collision_cooldown = 1.0  # שנייה
            self.speed *= 0.2  # האטה משמעותית
            damage = 20
        elif obstacle_type == "tree":
            self.collision_cooldown = 1.5  # שנייה וחצי
            self.speed *= 0.1  # כמעט עצירה מוחלטת
            damage = 30
        elif obstacle_type == "cone":
            self.collision_cooldown = 0.5  # חצי שנייה
            self.speed *= 0.5  # האטה קלה
            damage = 5
        elif obstacle_type == "puddle":
            self.collision_cooldown = 0.8  # 0.8 שניות
            # בשלולית המכונית תחליק (כיוון אקראי מעט)
            slip_angle = random.uniform(-20, 20)
            self.rotation += slip_angle
            damage = 0  # אין נזק בשלולית
        else:
            self.collision_cooldown = 0.7  # ברירת מחדל
            self.speed *= 0.3
            damage = 10
        
        # עדכון נזק למכונית
        self.take_damage(damage)
        
        # סימון המכונית כמתנגשת
        self.is_colliding = True
        self.last_collision_time = time.time()
    
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
    
    def set_world_dimensions(self, width, height):
        """
        קביעת ממדי העולם הוירטואלי
        
        Args:
            width: רוחב העולם
            height: גובה העולם
        """
        self.world_width = width
        self.world_height = height
        
    def set_screen_dimensions(self, width, height):
        """
        Set the screen dimensions for boundary checking
        
        Args:
            width: Screen width
            height: Screen height
        """
        self.screen_width = width
        self.screen_height = height