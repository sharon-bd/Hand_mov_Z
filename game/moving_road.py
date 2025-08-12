#!/usr/bin/env python
"""
Enhanced moving road generator - supports lateral car offset

This module implements a moving road background with lane markings only.
"""

import pygame
import math
import random

class MovingRoadGenerator:
    """Moving road generator with support for car lateral offset"""
    
    def __init__(self, screen_width, screen_height):
        """
        Initialize the moving road generator
        
        Args:
            screen_width: Game screen width
            screen_height: Game screen height
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Road parameters
        self.road_width = 300
        self.lane_width = 10
        self.lane_length = 50
        self.lane_gap = 20
        
        # Colors
        self.road_color = (80, 80, 80)
        self.lane_color = (240, 240, 240)
        self.grass_color = (0, 100, 0)
        
        # Scroll state
        self.scroll_y = 0
        self.scroll_x = 0
        self.speed = 0.0
        
        # === New lateral offset support ===
        self.car_road_offset = 0.0      # Car offset from centerline
        self.road_visual_offset = 0.0   # Visual offset of the road
        self.offset_smoothing = 0.15    # Road movement smoothing
        
        print("✅ MovingRoadGenerator initialized with lateral offset support")
    
    def set_car_road_offset(self, offset):
        """
        Set car offset from road centerline
        
        Args:
            offset: Offset in pixels (-150 to +150)
        """
        self.car_road_offset = offset
    
    def update(self, rotation, speed, dt, car_road_offset=None):
        """
        Update moving road state with lateral offset support
        
        Args:
            rotation: Car rotation in degrees
            speed: Normalized car speed (0.0 to 1.0)
            dt: Delta time in seconds
            car_road_offset: Car offset from centerline (optional)
        """
        # Save current speed
        self.speed = speed
        
        # עדכון היסט המכונית אם סופק
        if car_road_offset is not None:
            self.car_road_offset = car_road_offset
        
        # חישוב מהירות גלילה
        base_scroll_speed = 500 * speed
        
        # עדכון מיקום גלילה אנכי
        self.scroll_y += base_scroll_speed * dt
        
        # === עדכון היסט צידי של הכביש ===
        # הכביש זז בכיוון הפוך להיסט המכונית כדי ליצור אשליה של תנועה
        target_road_offset = -self.car_road_offset * 0.8  # 80% מההיסט
        
        # החלקה של תנועת הכביש למראה טבעי יותר
        offset_diff = target_road_offset - self.road_visual_offset
        self.road_visual_offset += offset_diff * self.offset_smoothing
        
        # עדכון גלילה אופקית מינימלית על סמך סיבוב
        steering_factor = rotation / 90.0
        steering_factor = max(-0.2, min(0.2, steering_factor))
        
        horizontal_speed = steering_factor * 15 * speed
        self.scroll_x += horizontal_speed * dt

    def draw(self, screen, world_offset_x=0, world_offset_y=0):
        """
        ציור הכביש הנע עם היסט צידי
        
        Args:
            screen: משטח Pygame לציור
            world_offset_x: היסט נוסף בעולם בכיוון X
            world_offset_y: היסט נוסף בעולם בכיוון Y
        """
        # מילוי רקע בצבע דשא
        screen.fill(self.grass_color)
        
        # === חישוב מיקום הכביש עם היסט צידי ===
        road_center_x = self.screen_width // 2 + self.road_visual_offset
        
        # ציור משטח הכביש הראשי עם היסט
        road_rect = pygame.Rect(
            road_center_x - self.road_width // 2,
            0,
            self.road_width,
            self.screen_height
        )
        
        # וידוא שהכביש נמצא בגבולות המסך (לפחות חלקית)
        if road_rect.right > 0 and road_rect.left < self.screen_width:
            pygame.draw.rect(screen, self.road_color, road_rect)
        
        # ציור קו מרכז נע עם היסט
        self._draw_center_line_with_offset(screen, road_center_x)
        
        # ציור אלמנטים נוספים של הכביש
        self._draw_road_markers_with_offset(screen, road_center_x)
    
    def _draw_center_line_with_offset(self, screen, center_x):
        """ציור קו מרכז נע עם היסט צידי"""
        line_width = 6
        
        # קו מרכז מקווקו נע
        dash_spacing = self.lane_length + self.lane_gap
        offset = int(-self.scroll_y) % dash_spacing  # כיוון הפוך לתנועת המכונית
        
        y = -offset
        while y < self.screen_height + dash_spacing:
            if y + self.lane_length > 0 and y < self.screen_height:
                dash_start = max(0, y)
                dash_end = min(self.screen_height, y + self.lane_length)
                
                # ציור רק אם הקו נמצא בגבולות המסך
                if 0 <= center_x - line_width // 2 <= self.screen_width + 50:
                    pygame.draw.rect(screen, self.lane_color,
                                   (center_x - line_width // 2, dash_start,
                                    line_width, dash_end - dash_start))
            
            y += dash_spacing
    
    def _draw_road_markers_with_offset(self, screen, road_center_x):
        """ציור סמני כביש עם היסט צידי"""
        # סמני קילומטר בצדי הכביש
        marker_spacing = 200
        marker_offset = int(-self.scroll_y) % marker_spacing
        
        y = -marker_offset
        marker_count = 0
        
        while y < self.screen_height + marker_spacing:
            if y > -50 and y < self.screen_height + 50:
                # סמן שמאלי
                left_marker_x = road_center_x - self.road_width // 2 - 20
                if -30 <= left_marker_x <= self.screen_width + 30:
                    self._draw_kilometer_marker(screen, left_marker_x, y, marker_count, "left")
                
                # סמן ימני
                right_marker_x = road_center_x + self.road_width // 2 + 20
                if -30 <= right_marker_x <= self.screen_width + 30:
                    self._draw_kilometer_marker(screen, right_marker_x, y, marker_count, "right")
                
                marker_count += 1
            
            y += marker_spacing
    
    def _draw_kilometer_marker(self, screen, x, y, number, side):
        """ציור סמן קילומטר יחיד"""
        # עמוד לבן
        pygame.draw.rect(screen, (255, 255, 255), (x - 3, y - 25, 6, 50))
        
        # רפלקטור צבעוני
        color = (255, 0, 0) if side == "left" else (0, 255, 0)
        pygame.draw.circle(screen, color, (int(x), int(y - 15)), 5)
        
        # מספר הסמן (כל 5 סמנים)
        if number % 5 == 0:
            font = pygame.font.Font(None, 20)
            text = font.render(str(number // 5), True, (255, 255, 255))
            text_rect = text.get_rect(center=(x, y + 20))
            screen.blit(text, text_rect)
    
    def get_debug_info(self):
        """קבלת מידע דיבוג"""
        return {
            'version': 'Enhanced Road with Lateral Offset 1.0',
            'scroll_y': self.scroll_y,
            'scroll_x': self.scroll_x,
            'speed': self.speed,
            'car_road_offset': self.car_road_offset,
            'road_visual_offset': self.road_visual_offset
        }
    
    def reset(self):
        """איפוס הכביש למצב התחלתי"""
        self.scroll_y = 0
        self.scroll_x = 0
        self.speed = 0.0
        self.car_road_offset = 0.0
        self.road_visual_offset = 0.0
        
        print("✅ MovingRoad נאופס עם תמיכה בהיסט צידי")
    
    def get_road_bounds(self):
        """קבלת גבולות הכביש לזיהוי התנגשויות"""
        road_center_x = self.screen_width // 2 + self.road_visual_offset
        left_bound = road_center_x - self.road_width // 2
        right_bound = road_center_x + self.road_width // 2
        
        return {
            'left': left_bound,
            'right': right_bound,
            'top': 0,
            'bottom': self.screen_height,
            'center': road_center_x,
            'width': self.road_width,
            'visual_offset': self.road_visual_offset
        }

# Export main class
__all__ = ['MovingRoadGenerator']

# יחידת בדיקה
if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("MovingRoad Test - עם היסט צידי")
    clock = pygame.time.Clock()
    
    road = MovingRoadGenerator(800, 600)
    running = True
    speed = 0.0
    car_offset = 0.0  # היסט המכונית
    
    print("🎮 בדיקת MovingRoad עם היסט צידי")
    print("השתמש בחצים UP/DOWN לשליטה במהירות")
    print("השתמש בחצים LEFT/RIGHT לשליטה בהיסט המכונית")
    
    while running:
        dt = clock.tick(60) / 1000.0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # טיפול בקלט
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            speed = min(1.0, speed + dt)
        elif keys[pygame.K_DOWN]:
            speed = max(0.0, speed - dt)
        else:
            speed *= 0.95  # האטה הדרגתית
        
        # שליטה בהיסט המכונית
        if keys[pygame.K_LEFT]:
            car_offset = max(-150, car_offset - 100 * dt)
        elif keys[pygame.K_RIGHT]:
            car_offset = min(150, car_offset + 100 * dt)
        else:
            car_offset *= 0.9  # חזרה הדרגתית למרכז
        
        # עדכון וציור
        road.update(0, speed, dt, car_offset)
        road.draw(screen)
        
        # ציור מידע דיבוג
        font = pygame.font.Font(None, 36)
        speed_text = font.render(f"מהירות: {speed:.2f}", True, (255, 255, 255))
        offset_text = font.render(f"היסט מכונית: {car_offset:.1f}", True, (255, 255, 255))
        
        screen.blit(speed_text, (10, 10))
        screen.blit(offset_text, (10, 50))
        
        # ציור מכונית דמה במרכז
        car_x = 400 + car_offset * 0.4  # 40% מההיסט מוצג
        car_y = 400
        pygame.draw.rect(screen, (50, 50, 200), (car_x - 20, car_y - 40, 40, 80))
        
        debug_info = road.get_debug_info()
        info_text = font.render(f"כביש עם היסט: {debug_info['version']}", True, (0, 255, 0))
        screen.blit(info_text, (10, 90))
        
        pygame.display.flip()
    
    pygame.quit()
    print("👋 בדיקה הסתיימה")