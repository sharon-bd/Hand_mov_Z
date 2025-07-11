#!/usr/bin/env python
"""
שיפור פיזיקת המכונית - המכונית מתרחקת ממרכז הכביש בזמן פנייה
"""

import math
import pygame
import time
import random

class Car:
    """
    מחלקת מכונית עם פיזיקה מתקדמת של פנייה
    """
    
    def __init__(self, x, y, width=40, height=80, screen_width=800, screen_height=600):
        # מיקום ומימדים
        self.x = x  # מיקום אמיתי בעולם
        self.y = y  # מיקום אמיתי בעולם
        self.screen_x = x  # תמיד במרכז המסך
        self.screen_y = y  # תמיד במרכז המסך
        self.width = width
        self.height = height
        
        # פרמטרי תנועה
        self.direction = 0.0  # -1.0 (שמאל) עד 1.0 (ימין)
        self.speed = 1.0      # 0.0 עד 1.0
        self.max_speed = 8.0  # מהירות מקסימלית בפיקסלים לפריים
        self.boost_multiplier = 1.5
        self.brake_deceleration = 0.4
        self.min_speed = 1.0
        
        # === פרמטרי פנייה משופרים ===
        self.steering_sensitivity = 2.0      # רגישות הגה (מוגברת)
        self.max_steering_angle = 30         # זווית הגה מקסימלית
        self.steering_return_factor = 0.08   # החזרה למרכז (מוחלשת)
        self.max_turn_rate = 60              # מעלות לשנייה מקסימליות
        self.steering_deadzone = 0.05        # אזור מת של ההגה (מוקטן)
        
        # === פיזיקת פנייה חדשה ===
        self.lateral_velocity = 0.0          # מהירות צידית
        self.lateral_acceleration = 0.0      # תאוצה צידית
        self.centrifugal_force = 0.0         # כוח צנטריפוגלי
        self.drift_factor = 0.15             # גורם החלקה בפנייה
        self.lateral_friction = 0.85         # חיכוך צידי
        
        # === היסט צידי מהכביש ===
        self.road_offset = 0.0               # היסט מקו האמצע של הכביש
        self.max_road_offset = 150           # מרחק מקסימלי מהכביש
        self.offset_acceleration = 0.0       # תאוצת היסט
        
        # גבולות עולם
        self.world_width = 2000
        self.world_height = 2000
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # מצבי התנגשות
        self.collision_cooldown = 0
        self.collision_flash = False
        self.last_collision_time = 0
        self.is_colliding = False
        
        # מצבים מיוחדים
        self.boost_active = False
        self.braking = False
        
        # מראה
        self.color = (50, 50, 200)
        self.show_indicators = True
        
        # תכונות פיזיקליות
        self.mass = 1000
        self.rotation = 0
        self.target_rotation = 0
        
        # זיהוי התנגשויות
        self.collision_points = []
        self.update_collision_points()
        
        # תכונות נוספות
        self.health = 100
        self.score = 0
        
        # היסטוריה לאפקט שובל
        self.position_history = []
        self.max_history = 20
        
        print("✅ מכונית אותחלה עם פיזיקת פנייה משופרת")
    
    def update(self, controls, dt):
        """
        עדכון מצב המכונית עם פיזיקת פנייה משופרת
        """
        # מונה סטטי לדיבוג
        if not hasattr(Car, '_debug_counter'):
            Car._debug_counter = 0
            Car._debug_enabled = True
            Car._last_debug_time = time.time()
        
        Car._debug_counter += 1
        current_time = time.time()
        if Car._debug_enabled and current_time - Car._last_debug_time > 3.0:
            Car._last_debug_time = current_time
            Car._debug_counter = 0
            brief_controls = {
                'steering': controls.get('steering', 0),
                'throttle': controls.get('throttle', 0),
                'braking': controls.get('braking', False),
                'boost': controls.get('boost', False),
                'gesture_name': controls.get('gesture_name', 'Unknown')
            }
            print(f"🚗 מכונית מקבלת בקרות: {brief_controls}")
            print(f"🚗 מצב נוכחי: מיקום=({self.x:.1f},{self.y:.1f}), מהירות={self.speed:.2f}, סיבוב={self.rotation:.1f}°, היסט_כביש={self.road_offset:.1f}")
        
        try:
            # חילוץ בקרות
            steering = float(controls.get('steering', 0.0))
            throttle = float(controls.get('throttle', 0.0))
            braking = bool(controls.get('braking', False))
            boost = bool(controls.get('boost', False))
            
            # נירמול ערכים
            steering = max(-1.0, min(1.0, steering))
            throttle = max(0.0, min(1.0, throttle))
            
            # יישום אזור מת
            if abs(steering) < self.steering_deadzone:
                steering = 0.0
            
            self.direction = steering
            
            # עדכון מהירות
            speed_change_rate = 0.1 if throttle > self.speed else 0.2
            self.speed = self.speed + (throttle - self.speed) * speed_change_rate
            
            # טיפול בבלימה
            if braking:
                self.speed = max(0.0, self.speed - self.brake_deceleration * dt)
                self.braking = True
            else:
                self.braking = (self.speed < 0.05 and throttle < 0.1)
            
            # חישוב תנועה
            movement_speed = self.max_speed * self.speed
            if boost:
                movement_speed *= self.boost_multiplier
            
            # === פיזיקת פנייה משופרת ===
            previous_rotation = self.rotation
            
            if self.speed > 0.05:
                # חישוב אפקט הגה עם רגישות מוגברת
                steering_effect = self.direction * self.steering_sensitivity
                
                # פקטור מהירות - פחות השפעה של מהירות על רגישות ההגה
                speed_factor = max(0.4, 1.0 - (self.speed * 0.8))
                steering_effect *= speed_factor
                
                # חישוב שינוי זווית מקסימלי
                max_angle_change = self.max_steering_angle * self.speed
                max_rate_limited_change = self.max_turn_rate * dt
                
                max_allowed_change = min(max_angle_change, max_rate_limited_change)
                
                # יישום שינוי סיבוב
                rotation_change = min(max_allowed_change, 
                                     max(-max_allowed_change, 
                                         steering_effect * max_allowed_change))
                
                self.rotation += rotation_change
                
                # === חישוב כוח צנטריפוגלי והיסט צידי ===
                if abs(rotation_change) > 0.1:  # רק בפנייה
                    # כוח צנטריפוגלי פרופורציונלי למהירות ולזווית הפנייה
                    self.centrifugal_force = abs(rotation_change) * self.speed * 2.5
                    
                    # תאוצה צידית - המכונית "נדחפת" החוצה בפנייה
                    direction_multiplier = 1 if rotation_change > 0 else -1
                    self.lateral_acceleration = self.centrifugal_force * direction_multiplier * 0.8
                    
                    # עדכון מהירות צידית
                    self.lateral_velocity += self.lateral_acceleration * dt
                    
                    # הגבלת מהירות צידית מקסימלית
                    max_lateral_velocity = self.speed * 3.0
                    self.lateral_velocity = max(-max_lateral_velocity, 
                                               min(max_lateral_velocity, self.lateral_velocity))
                else:
                    # אין פנייה - החזרה הדרגתית למרכז
                    self.centrifugal_force = 0.0
                    self.lateral_acceleration = 0.0
                
                # החלת חיכוך צידי - מעט הפחתה של המהירות הצידית
                self.lateral_velocity *= self.lateral_friction
                
                # עדכון היסט מהכביש
                self.road_offset += self.lateral_velocity * dt
                
                # === החזרה הדרגתית למרכז הכביש (מוחלשת) ===
                if abs(self.direction) < 0.6:  # רק כשלא פונים בחדות
                    center_return_force = -self.road_offset * 0.3 * dt  # כוח חזרה חלש יותר
                    self.road_offset += center_return_force
                
                # הגבלת היסט מקסימלי
                self.road_offset = max(-self.max_road_offset, 
                                     min(self.max_road_offset, self.road_offset))
                
                # טיפול בהיסטוריית סיבוב למניעת ספינינג
                self.rotation_history = getattr(self, 'rotation_history', [])
                self.rotation_history.append(self.rotation)
                if len(self.rotation_history) > 20:
                    self.rotation_history.pop(0)
                
                # החזרה למרכז כשלא פונים
                if abs(self.direction) < 0.3:
                    self.target_rotation = round(self.rotation / 45) * 45
                    angle_diff = (self.target_rotation - self.rotation) * self.steering_return_factor
                    self.rotation += angle_diff * dt
                
                self.last_rotation = self.rotation
            
            # שמירה על סיבוב בטווח 0-360
            self.rotation %= 360
            
            # חישוב וקטור תנועה
            rad = math.radians(self.rotation)
            distance = movement_speed * dt
            dx = math.sin(rad) * distance
            dy = -math.cos(rad) * distance
            
            # עדכון מיקום אמיתי בעולם
            new_x = self.x + dx
            new_y = self.y + dy
            
            # בדיקת גבולות עולם
            hit_boundary = False
            
            if new_x < 0:
                new_x = 0
                hit_boundary = True
            elif new_x > self.world_width:
                new_x = self.world_width
                hit_boundary = True
                
            if new_y < 0:
                new_y = 0
                hit_boundary = True
            elif new_y > self.world_height:
                new_y = self.world_height
                hit_boundary = True
            
            if hit_boundary:
                self.speed *= 0.4
                self.lateral_velocity *= 0.5  # הפחתת מהירות צידית בפגיעה בגבול
                
                bounce_factor = -0.2
                if new_x == 0 or new_x == self.world_width:
                    self.x += dx * bounce_factor
                if new_y == 0 or new_y == self.world_height:
                    self.y += dy * bounce_factor
            
            # עדכון מיקום
            self.x = new_x
            self.y = new_y
            
            # עדכון מצב התנגשות
            if self.collision_cooldown > 0:
                self.collision_cooldown -= dt
                self.collision_flash = int(self.collision_cooldown * 10) % 2 == 0
            else:
                self.collision_flash = False
                self.is_colliding = False
            
            # המכונית תישאר במרכז המסך עם היסט צידי
            screen_center_x = self.screen_width // 2
            screen_center_y = self.screen_height - 100
            
            # === יישום ההיסט הצידי על מיקום המסך ===
            self.screen_x = screen_center_x + self.road_offset * 0.6  # 60% מההיסט למסך
            self.screen_y = screen_center_y
            
            # עדכון נקודות התנגשות
            self.update_collision_points()
            
            # שמירת מיקום להיסטוריה
            if distance > 0:
                self.position_history.append((self.x, self.y))
                if len(self.position_history) > self.max_history:
                    self.position_history.pop(0)
        
        except Exception as e:
            print(f"שגיאה בעדכון המכונית: {e}")
            import traceback
            traceback.print_exc()
    
    def draw(self, screen, offset_x=0, offset_y=0):
        """
        ציור המכונית על המסך עם היסט צידי
        """
        try:
            # חישוב מיקום המסך עם היסט צידי
            screen_center_x = self.screen_width // 2
            screen_center_y = self.screen_height // 2
            
            # המכונית מוצגת במיקום שמשקף את ההיסט מהכביש
            display_x = screen_center_x + self.road_offset * 0.4  # 40% מההיסט מוצג
            display_y = screen_center_y
            
            # ציור אפקט שובל אם זזה
            if self.position_history and self.speed > 0.1:
                for i, (pos_x, pos_y) in enumerate(self.position_history):
                    # חישוב מיקום מסך על סמך מיקום עולם
                    trail_screen_x = display_x - (self.x - pos_x)
                    trail_screen_y = display_y - (self.y - pos_y)
                    
                    alpha = int(255 * (i / len(self.position_history)))
                    trail_size = max(3, int(self.width * 0.3 * (i / len(self.position_history))))
                    pygame.draw.circle(
                        screen,
                        (*self.color[:3], alpha),
                        (int(trail_screen_x), int(trail_screen_y)),
                        trail_size
                    )
            
            # יצירת משטח מכונית מסובב
            car_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            
            # בחירת צבע לפי מצב
            color = self.color
            if self.collision_flash:
                color = (255, 0, 0)
            elif self.boost_active:
                color = (0, 150, 255)
            elif self.braking:
                color = (200, 50, 50)
            
            # ציור גוף המכונית
            pygame.draw.rect(
                car_surface, 
                color,
                (0, 0, self.width, self.height),
                0, 10
            )
            
            # ציור שמשה קדמית
            windshield_width = self.width * 0.7
            windshield_height = self.height * 0.3
            pygame.draw.rect(
                car_surface,
                (150, 220, 255),
                (
                    (self.width - windshield_width) / 2,
                    self.height * 0.15,
                    windshield_width,
                    windshield_height
                ),
                0, 5
            )
            
            # ציור פנסים קדמיים
            light_size = self.width // 5
            pygame.draw.circle(
                car_surface,
                (255, 255, 200),
                (self.width // 4, light_size),
                light_size // 2
            )
            pygame.draw.circle(
                car_surface,
                (255, 255, 200),
                (self.width - self.width // 4, light_size),
                light_size // 2
            )
            
            # ציור אורות בלימה
            if self.braking:
                # אור שמאלי מוגבר
                pygame.draw.circle(
                    car_surface,
                    (255, 30, 30),
                    (self.width // 4, self.height - light_size),
                    light_size // 2 + 2
                )
                # אפקט זוהר
                pygame.draw.circle(
                    car_surface,
                    (255, 100, 100, 150),
                    (self.width // 4, self.height - light_size),
                    light_size // 2 + 5
                )
                
                # אור ימני מוגבר
                pygame.draw.circle(
                    car_surface,
                    (255, 30, 30),
                    (self.width - self.width // 4, self.height - light_size),
                    light_size // 2 + 2
                )
                pygame.draw.circle(
                    car_surface,
                    (255, 100, 100, 150),
                    (self.width - self.width // 4, self.height - light_size),
                    light_size // 2 + 5
                )
            
            # ציור אפקט בוסט
            if self.boost_active:
                flame_points = [
                    (self.width // 2, self.height),
                    (self.width // 2 - self.width // 4, self.height + self.height // 3),
                    (self.width // 2 + self.width // 4, self.height + self.height // 3)
                ]
                pygame.draw.polygon(car_surface, (255, 165, 0), flame_points)
                
                inner_flame_points = [
                    (self.width // 2, self.height),
                    (self.width // 2 - self.width // 8, self.height + self.height // 4),
                    (self.width // 2 + self.width // 8, self.height + self.height // 4)
                ]
                pygame.draw.polygon(car_surface, (255, 255, 0), inner_flame_points)
            
            # סיבוב משטח המכונית
            rotated_car = pygame.transform.rotate(car_surface, -self.rotation)
            
            # מיקום המכונית המסובבת
            rotated_rect = rotated_car.get_rect(center=(display_x, display_y))
            
            # ציור המכונית המסובבת
            screen.blit(rotated_car, rotated_rect)
            
            # ציור אינדיקטורים לדיבוג
            if self.show_indicators:
                # אינדיקטור כיוון
                indicator_length = 50
                dx = math.sin(math.radians(self.rotation)) * indicator_length
                dy = -math.cos(math.radians(self.rotation)) * indicator_length
                pygame.draw.line(
                    screen,
                    (0, 255, 0),
                    (display_x, display_y),
                    (display_x + dx, display_y + dy),
                    2
                )
                
                # אינדיקטור מהירות
                pygame.draw.rect(
                    screen,
                    (0, 255, 0) if not self.boost_active else (255, 165, 0),
                    (
                        display_x - self.width//2 - 20,
                        display_y - self.height//2,
                        10,
                        self.height * self.speed
                    )
                )
                
                # === אינדיקטור היסט צידי חדש ===
                # קו שמראה את ההיסט מקו האמצע של הכביש
                road_center_x = screen_center_x
                pygame.draw.line(
                    screen,
                    (255, 255, 0),  # צהוב
                    (road_center_x, display_y - 30),
                    (display_x, display_y - 30),
                    3
                )
                
                # נקודה שמסמנת את מרכז הכביש
                pygame.draw.circle(
                    screen,
                    (255, 255, 0),
                    (road_center_x, display_y - 30),
                    5
                )
                
        except Exception as e:
            print(f"❌ שגיאה בציור המכונית: {e}")
            import traceback
            traceback.print_exc()
    
    def update_collision_points(self):
        """עדכון נקודות זיהוי התנגשויות"""
        rad = math.radians(self.rotation)
        sin_rot = math.sin(rad)
        cos_rot = math.cos(rad)
        
        half_width = self.width // 2
        half_height = self.height // 2
        
        points = [
            (0, -half_height),  # חזית
            (-half_width, -half_height),  # חזית שמאל
            (half_width, -half_height),  # חזית ימין
            (0, half_height),  # אחור
            (-half_width, half_height),  # אחור שמאל
            (half_width, half_height),  # אחור ימין
            (0, 0)  # מרכז
        ]
        
        # סיבוב נקודות והוספת מיקום המכונית
        self.collision_points = []
        for px, py in points:
            # סיבוב נקודה
            rx = px * cos_rot - py * sin_rot
            ry = px * sin_rot + py * cos_rot
            
            # הוספת מיקום המכונית
            self.collision_points.append((self.x + rx, self.y + ry))
    
    def check_collision(self, obstacle):
        """בדיקת התנגשות עם מכשול"""
        obstacle_rect = pygame.Rect(
            obstacle['x'] - obstacle['width']//2,
            obstacle['y'] - obstacle['height']//2,
            obstacle['width'],
            obstacle['height']
        )
        
        for point in self.collision_points:
            if obstacle_rect.collidepoint(point):
                return True
        
        return False
    
    def handle_obstacle_collision(self, obstacle_type=None):
        """טיפול בהתנגשות עם מכשול"""
        if self.collision_cooldown > 0:
            return
        
        # קביעת השפעת ההתנגשות
        if obstacle_type == "rock":
            self.collision_cooldown = 1.0
            self.speed *= 0.2
            damage = 20
        elif obstacle_type == "tree":
            self.collision_cooldown = 1.5
            self.speed *= 0.1
            damage = 30
        elif obstacle_type == "cone":
            self.collision_cooldown = 0.5
            self.speed *= 0.5
            damage = 5
        elif obstacle_type == "puddle":
            self.collision_cooldown = 0.8
            # בשלולית - המכונית תחליק (זווית אקראית)
            slip_angle = random.uniform(-20, 20)
            self.rotation += slip_angle
            # הוספת מהירות צידית אקראית
            self.lateral_velocity += random.uniform(-2, 2)
            damage = 0
        else:
            self.collision_cooldown = 0.7
            self.speed *= 0.3
            damage = 10
        
        # עדכון נזק
        self.take_damage(damage)
        
        # סימון התנגשות
        self.is_colliding = True
        self.last_collision_time = time.time()
    
    def take_damage(self, amount):
        """הפחתת בריאות המכונית"""
        self.health -= amount
        self.health = max(0, self.health)
        return self.health <= 0
    
    def repair(self, amount):
        """תיקון המכונית"""
        self.health += amount
        self.health = min(100, self.health)
    
    def add_score(self, points):
        """הוספת נקודות"""
        self.score += points
        
    def clear_trail(self):
        """ניקוי שובל המכונית"""
        self.position_history = []
        
    def reset_state(self):
        """איפוס מצב המכונית"""
        self.speed = 0.0
        self.direction = 0.0
        self.health = 100
        self.boost_active = False
        self.braking = False
        self.road_offset = 0.0  # איפוס היסט הכביש
        self.lateral_velocity = 0.0  # איפוס מהירות צידית
        self.clear_trail()
    
    def set_world_dimensions(self, width, height):
        """קביעת ממדי העולם"""
        self.world_width = width
        self.world_height = height
        
    def set_screen_dimensions(self, width, height):
        """קביעת ממדי המסך"""
        self.screen_width = width
        self.screen_height = height
    
    def get_road_position_info(self):
        """מידע על מיקום המכונית ביחס לכביש"""
        return {
            'road_offset': self.road_offset,
            'lateral_velocity': self.lateral_velocity,
            'centrifugal_force': self.centrifugal_force,
            'max_offset': self.max_road_offset,
            'offset_percentage': (self.road_offset / self.max_road_offset) * 100
        }

# דוגמה לשימוש עם מידע על מיקום הכביש
def example_usage():
    """דוגמה לשימוש במחלקה המשופרת"""
    car = Car(400, 300)
    
    # דמיון של לולאת משחק
    controls = {
        'steering': 0.8,  # פנייה חדה ימינה
        'throttle': 0.7,  # מהירות בינונית
        'braking': False,
        'boost': False
    }
    
    # עדכון המכונית
    car.update(controls, 0.016)  # 60 FPS
    
    # קבלת מידע על מיקום
    road_info = car.get_road_position_info()
    print(f"היסט מהכביש: {road_info['road_offset']:.1f} פיקסלים")
    print(f"אחוז היסט: {road_info['offset_percentage']:.1f}%")

if __name__ == "__main__":
    example_usage()