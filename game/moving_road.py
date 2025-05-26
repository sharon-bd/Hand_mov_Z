"""
Moving Road Module for Hand Gesture Car Control Game

This module implements a moving road background that responds to car movement,
creating the illusion that the car is moving while actually staying in the center
of the screen.
"""

import pygame
import math
import random
import numpy as np

class MovingRoadGenerator:
    """Class that generates a moving road based on car movement"""
    
    def __init__(self, screen_width, screen_height):
        """
        Initialize the moving road generator
        
        Args:
            screen_width: Width of the game screen
            screen_height: Height of the game screen
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
        
        # Scrolling state
        self.scroll_y = 0
        self.scroll_x = 0
        self.rotation = 0
        
        # Scroll speed parameters
        self.max_scroll_speed = 500  # Maximum scroll speed in pixels per second
        self.car_max_speed = 300     # Car's maximum speed in pixels per second
        
        # Generate random road elements
        self.road_elements = []
        self._generate_road_elements(50)  # Generate 50 random elements
        
        # Add track markings that show clear movement
        self.track_markings = []
        self._generate_track_markings(100)  # Generate 100 track markings
        
        # Add moving particles for a sense of speed
        self.particles = []
        self._generate_particles(50)  # Generate 50 particles

        # הוספת חלקיקים לאפקט מהירות
        self.speed_particles = []
        for _ in range(30):
            self.speed_particles.append({
                'x': random.randint(0, screen_width),
                'y': random.randint(0, screen_height),
                'speed': random.uniform(1.5, 3.0),
                'size': random.randint(1, 2)
            })
    
    def _generate_road_elements(self, count):
        """
        Generate random elements along the road sides
        
        Args:
            count: Number of elements to generate
        """
        self.road_elements = []
        
        # Original random elements (rocks, bushes, trees)
        for _ in range(count):
            side = random.choice([-1, 1])  # Left or right side
            x_offset = side * (self.road_width / 2 + random.randint(20, 100))
            y = random.randint(-500, self.screen_height + 500)
            
            element_type = random.choice(['rock', 'bush', 'tree'])
            
            if element_type == 'rock':
                size = random.randint(5, 15)
                color = (120, 120, 120)  # Gray
            elif element_type == 'bush':
                size = random.randint(10, 20)
                color = (0, 120, 0)  # Green
            else:  # tree
                size = random.randint(15, 25)
                color = (0, 80, 0)  # Dark green
            
            self.road_elements.append({
                'x': self.screen_width / 2 + x_offset,
                'y': y,
                'size': size,
                'color': color,
                'type': element_type
            })
        
        # הוספת אובייקטים צבעוניים להמחשת תזוזה
        for i in range(20):
            self.road_elements.append({
                'x': self.screen_width / 2 - self.road_width / 2 - 60 + random.randint(-20, 20),
                'y': random.randint(-500, self.screen_height + 2000),
                'size': random.randint(18, 28),
                'color': (0, 180, 0),
                'type': 'tree'
            })
            self.road_elements.append({
                'x': self.screen_width / 2 + self.road_width / 2 + 60 + random.randint(-20, 20),
                'y': random.randint(-500, self.screen_height + 2000),
                'size': random.randint(10, 18),
                'color': (200, 40, 40),
                'type': 'pole'
            })
        
        marker_spacing = 200
        road_length = self.screen_height * 3
        num_markers = int(road_length / marker_spacing)
        
        for i in range(num_markers):
            self.road_elements.append({
                'x': self.screen_width / 2 - self.road_width / 2 - 30,
                'y': -500 + i * marker_spacing,
                'size': 6,
                'color': (255, 255, 255),
                'type': 'distance_marker',
                'side': 'left',
                'number': i
            })
            
            self.road_elements.append({
                'x': self.screen_width / 2 + self.road_width / 2 + 30,
                'y': -500 + i * marker_spacing,
                'size': 6,
                'color': (255, 255, 255),
                'type': 'distance_marker',
                'side': 'right',
                'number': i
            })
        
        fence_spacing = 50
        fence_length = self.screen_height * 3
        num_fence_posts = int(fence_length / fence_spacing)
        
        for i in range(num_fence_posts):
            self.road_elements.append({
                'x': self.screen_width / 2 - self.road_width / 2 - 45,
                'y': -500 + i * fence_spacing,
                'size': 4,
                'color': (120, 100, 80),
                'type': 'fence_post'
            })
            
            self.road_elements.append({
                'x': self.screen_width / 2 + self.road_width / 2 + 45,
                'y': -500 + i * fence_spacing,
                'size': 4,
                'color': (120, 100, 80),
                'type': 'fence_post'
            })
        
        for i in range(5):
            self.road_elements.append({
                'x': self.screen_width / 2 - self.road_width / 2 - 80,
                'y': -200 + i * 800,
                'size': 20,
                'color': (220, 220, 220),
                'type': 'milestone',
                'number': i+1
            })
            
            self.road_elements.append({
                'x': self.screen_width / 2 + self.road_width / 2 + 80,
                'y': -200 + i * 800 + 400,
                'size': 20,
                'color': (220, 220, 220),
                'type': 'milestone',
                'number': i+1
            })
        
        for i in range(10):
            self.road_elements.append({
                'x': self.screen_width / 2 - self.road_width / 2 - 70 + random.randint(-10, 10),
                'y': -300 + i * 600 + random.randint(-50, 50),
                'size': 15,
                'color': (255, 0, 0),
                'type': 'stop_sign'
            })
            
            self.road_elements.append({
                'x': self.screen_width / 2 + self.road_width / 2 + 70 + random.randint(-10, 10),
                'y': -600 + i * 600 + random.randint(-50, 50),
                'size': 14,
                'color': (255, 255, 255),
                'type': 'speed_sign'
            })
        
        car_colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0), (255, 165, 0)]
        for i in range(8):
            side = random.choice([-1, 1])
            self.road_elements.append({
                'x': self.screen_width / 2 + side * (self.road_width / 2 + 50 + random.randint(0, 30)),
                'y': -400 + i * 800 + random.randint(-100, 100),
                'size': 20,
                'color': random.choice(car_colors),
                'type': 'parked_car',
                'rotation': random.randint(0, 359) if random.random() < 0.3 else 0
            })
        
        building_colors = [(100, 100, 100), (120, 100, 80), (80, 80, 100), (70, 90, 70)]
        for i in range(15):
            side = random.choice([-1, 1])
            distance = random.randint(120, 300)
            height = random.randint(40, 100)
            width = random.randint(40, 80)
            
            self.road_elements.append({
                'x': self.screen_width / 2 + side * (self.road_width / 2 + distance),
                'y': -800 + i * 400 + random.randint(-100, 100),
                'width': width,
                'height': height,
                'color': random.choice(building_colors),
                'type': 'building'
            })
        
        billboard_texts = ["SALE!", "EAT AT JOE'S", "GAME ON", "DRIVE SAFE", "NEW PHONES"]
        for i in range(6):
            side = random.choice([-1, 1])
            self.road_elements.append({
                'x': self.screen_width / 2 + side * (self.road_width / 2 + 100),
                'y': -700 + i * 700 + random.randint(-50, 50),
                'size': 25,
                'width': 80,
                'height': 40,
                'color': (200, 200, 200),
                'text': random.choice(billboard_texts),
                'type': 'billboard'
            })
    
    def _generate_track_markings(self, count):
        """
        Generate visual markings on the road to emphasize motion
        
        Args:
            count: Number of markings to generate
        """
        self.track_markings = []
        
        # Create track markings distributed along the entire length of the road
        for i in range(count):
            y_pos = -1000 + i * 100  # Distribute vertically with regular spacing
            
            # Create "mile marker" style markings on both sides
            self.track_markings.append({
                'x': self.screen_width / 2 - self.road_width / 2 - 15,
                'y': y_pos,
                'size': 8,
                'type': 'marker',
                'number': i // 10,  # Number every 10 markers
                'side': 'left'
            })
            
            self.track_markings.append({
                'x': self.screen_width / 2 + self.road_width / 2 + 15,
                'y': y_pos,
                'size': 8,
                'type': 'marker',
                'number': i // 10,
                'side': 'right'
            })

    def _generate_particles(self, count):
        """
        Generate particles for dynamic motion effects
        
        Args:
            count: Number of particles to generate
        """
        self.particles = []
        
        for i in range(count):
            # Create particles with random properties for a dynamic effect
            self.particles.append({
                'x': random.randint(0, self.screen_width),
                'y': random.randint(-1000, self.screen_height + 1000),
                'size': random.randint(1, 3),
                'speed': random.uniform(0.3, 1.5),  # Particles move at different speeds
                'color': (220, 220, 220)  # Light gray
            })

    def update(self, rotation, speed, dt):
        """
        Update the moving road state
        
        Args:
            rotation: Car rotation in degrees
            speed: Normalized car speed (0.0 to 1.0)
            dt: Time delta in seconds
        """
        if random.random() < 0.01:
            print(f"MovingRoad update: direction={rotation}, speed={speed}, dt={dt}")
        
        base_scroll_speed = self.max_scroll_speed * speed * (self.car_max_speed / 300.0)
        
        angle_rad = math.radians(rotation)
        
        # Update internal scroll values
        self.scroll_x += base_scroll_speed * math.sin(angle_rad) * dt
        self.scroll_y += base_scroll_speed * math.cos(angle_rad) * dt
        
        # Update car's visual rotation if the road itself is meant to rotate
        # self.rotation = rotation # Currently, self.rotation seems to stay 0. This is fine if rotation is handled by scroll vectors.
        
        # שמירת מהירות לשימוש בחלקיקים ובגריד
        self.speed = speed
        
        # Update particles positions based on car movement
        for particle in self.particles:
            # Move particles opposite to car direction at varying speeds
            particle['y'] += base_scroll_speed * particle['speed'] * math.cos(angle_rad) * dt
            particle['x'] -= base_scroll_speed * particle['speed'] * math.sin(angle_rad) * dt
            
            # Wrap around screen when they go out of bounds
            wrap_margin = 500
            if particle['y'] > self.screen_height + wrap_margin:
                particle['y'] = -wrap_margin
                particle['x'] = random.randint(0, self.screen_width)
            elif particle['y'] < -wrap_margin:
                particle['y'] = self.screen_height + wrap_margin
                particle['x'] = random.randint(0, self.screen_width)
                
            if particle['x'] > self.screen_width + wrap_margin:
                particle['x'] = -wrap_margin
                particle['y'] = random.randint(-wrap_margin, self.screen_height + wrap_margin)
            elif particle['x'] < -wrap_margin:
                particle['x'] = self.screen_width + wrap_margin
                particle['y'] = random.randint(-wrap_margin, self.screen_height + wrap_margin)

        # עדכון חלקיקי המהירות
        for particle in self.speed_particles:
            # תנועת החלקיקים בכיוון ההפוך לתנועה
            particle['y'] += base_scroll_speed * particle['speed'] * dt
            # איפוס חלקיק שיצא מהמסך
            if particle['y'] > self.screen_height:
                particle['y'] = -10
                particle['x'] = random.randint(0, self.screen_width)

    def draw(self, screen, world_offset_x=0, world_offset_y=0):
        """
        ציור המסלול הנע על המסך
        
        Args:
            screen: משטח Pygame לציור
            world_offset_x: היסט בעולם בכיוון x
            world_offset_y: היסט בעולם בכיוון y
        """
        # מילוי הרקע בצבע דשא
        screen.fill(self.grass_color)
        
             

        
        # יצירת משטח זמני למסלול שניתן לסובב
        road_surface = pygame.Surface((self.screen_width * 3, self.screen_height * 3), pygame.SRCALPHA)
        
        # נקודת המרכז של משטח המסלול
        center_x = road_surface.get_width() // 2
        center_y = road_surface.get_height() // 2
        
        # ציור המסלול הבסיסי
        road_rect = pygame.Rect(
            center_x - self.road_width // 2,
            0,
            self.road_width,
            road_surface.get_height()
        )
        pygame.draw.rect(road_surface, self.road_color, road_rect)
        
        # ציור סימוני נתיבים עם התאמה לגלילה
        for lane in range(2):  # 2 קווי נתיב ל-3 נתיבים
            lane_x = center_x - self.road_width // 2 + self.road_width * (lane + 1) // 3
            # Use internal scroll_y for lane markings
            y_offset = int(self.scroll_y) % (self.lane_length + self.lane_gap)
            for y in range(int(-y_offset), road_surface.get_height(), self.lane_length + self.lane_gap):
                pygame.draw.rect(
                    road_surface,
                    self.lane_color,
                    (lane_x - self.lane_width // 2, y, self.lane_width, self.lane_length)
                )
        
        # ===== הסרת קווי הגריד =====
        # (הסרתי את כל הקוד של ציור הגריד)
        
        # Road elements can still use world_offset_x/y for parallax if needed
        # The element_offset calculation below uses world_offset_x/y for parallax.
        # This is separate from the grid's scrolling.
        element_world_offset_x = int(self.scroll_x + world_offset_x * 1.5)
        element_world_offset_y = int(self.scroll_y + world_offset_y * 1.5)
        
        # הוספת רעש דיבוג מדי פעם להראות את ערכי ההיסטים
        if random.random() < 0.01:  # 1% מהפריימים
            print(f"Debug: Scroll XY=({self.scroll_x:.1f}, {self.scroll_y:.1f}), World offset=({world_offset_x:.1f}, {world_offset_y:.1f})")
            print(f"Debug: Grid offset=({grid_offset_x:.1f}, {grid_offset_y:.1f})")
        
        for element in self.road_elements:
            # המרת קואורדינטות למערכת הקואורדינטות של משטח המסלול
            base_x = center_x + (element['x'] - self.screen_width / 2)
            base_y = center_y + (element['y'] - self.screen_height / 2)
            
            # Apply scrolling and parallax for road elements
            # Note: Using element_world_offset_x/y which includes world_offset_x/y for parallax
            current_element_offset_x = element_world_offset_x % road_surface.get_width()
            current_element_offset_y = element_world_offset_y % road_surface.get_height()

            movement_multiplier = 1.2
            element_x = (base_x - current_element_offset_x * movement_multiplier) % road_surface.get_width()
            element_y = (base_y - current_element_offset_y * movement_multiplier) % road_surface.get_height()
            
            # ציור האלמנט בהתאם לסוג
            if element['type'] == 'rock':
                pygame.draw.circle(
                    road_surface,
                    element['color'],
                    (int(element_x), int(element_y)), 
                    element['size']
                )
            elif element['type'] == 'bush':
                pygame.draw.circle(
                    road_surface,
                    element['color'],
                    (int(element_x), int(element_y)),
                    element['size']
                )
            elif element['type'] == 'tree':
                trunk_width = element['size'] // 3
                trunk_height = element['size'] * 2
                pygame.draw.rect(
                    road_surface,
                    (101, 67, 33),
                    (int(element_x - trunk_width // 2), 
                     int(element_y - trunk_height // 2),
                     trunk_width, 
                     trunk_height)
                )
                pygame.draw.circle(
                    road_surface,
                    element['color'],
                    (int(element_x), int(element_y - trunk_height // 2)),
                    element['size']
                )
            elif element['type'] == 'pole':
                pygame.draw.circle(
                    road_surface,
                    element['color'],
                    (int(element_x), int(element_y)),
                    element['size']
                )
            elif element['type'] == 'distance_marker':
                marker_height = element['size'] * 5
                pygame.draw.rect(
                    road_surface,
                    element['color'],
                    (int(element_x - element['size']), 
                     int(element_y - marker_height / 2),
                     element['size'] * 2, 
                     marker_height)
                )
                
                pygame.draw.rect(
                    road_surface,
                    (0, 0, 0),
                    (int(element_x - element['size']), 
                     int(element_y - marker_height / 2),
                     element['size'] * 2, 
                     5)
                )
            
            elif element['type'] == 'fence_post':
                post_height = element['size'] * 4
                pygame.draw.rect(
                    road_surface,
                    element['color'],
                    (int(element_x - element['size'] / 2), 
                     int(element_y - post_height / 2),
                     element['size'], 
                     post_height)
                )
                
                pygame.draw.line(
                    road_surface,
                    element['color'],
                    (int(element_x - element['size'] * 3), int(element_y - post_height / 4)),
                    (int(element_x + element['size'] * 3), int(element_y - post_height / 4)),
                    2
                )
            
            elif element['type'] == 'milestone':
                pygame.draw.circle(
                    road_surface,
                    element['color'],
                    (int(element_x), int(element_y)),
                    element['size']
                )
                
                font_size = max(10, int(element['size'] * 1.2))
                milestone_font = pygame.font.SysFont(None, font_size)
                number_text = milestone_font.render(str(element['number']), True, (0, 0, 0))
                number_rect = number_text.get_rect(center=(int(element_x), int(element_y)))
                road_surface.blit(number_text, number_rect)
            
            elif element['type'] == 'stop_sign':
                pygame.draw.polygon(
                    road_surface,
                    element['color'],
                    [
                        (int(element_x), int(element_y - element['size'])),
                        (int(element_x + element['size'] * 0.7), int(element_y - element['size'] * 0.7)),
                        (int(element_x + element['size']), int(element_y)),
                        (int(element_x + element['size'] * 0.7), int(element_y + element['size'] * 0.7)),
                        (int(element_x), int(element_y + element['size'])),
                        (int(element_x - element['size'] * 0.7), int(element_y + element['size'] * 0.7)),
                        (int(element_x - element['size']), int(element_y)),
                        (int(element_x - element['size'] * 0.7), int(element_y - element['size'] * 0.7))
                    ]
                )
                stop_font = pygame.font.SysFont(None, int(element['size'] * 1.5))
                stop_text = stop_font.render("STOP", True, (255, 255, 255))
                stop_rect = stop_text.get_rect(center=(int(element_x), int(element_y)))
                road_surface.blit(stop_text, stop_rect)
                
                pygame.draw.rect(
                    road_surface,
                    (100, 100, 100),
                    (int(element_x - 2), int(element_y + element['size']), 4, element['size'] * 3)
                )
                
            elif element['type'] == 'speed_sign':
                pygame.draw.circle(
                    road_surface,
                    element['color'],
                    (int(element_x), int(element_y)),
                    element['size']
                )
                pygame.draw.circle(
                    road_surface,
                    (255, 0, 0),
                    (int(element_x), int(element_y)),
                    element['size'],
                    2
                )
                
                speed_font = pygame.font.SysFont(None, int(element['size'] * 1.8))
                speed_text = speed_font.render(str(random.choice([30, 50, 70, 90])), True, (0, 0, 0))
                speed_rect = speed_text.get_rect(center=(int(element_x), int(element_y)))
                road_surface.blit(speed_text, speed_rect)
                
                pygame.draw.rect(
                    road_surface,
                    (100, 100, 100),
                    (int(element_x - 2), int(element_y + element['size']), 4, element['size'] * 3)
                )
                
            elif element['type'] == 'parked_car':
                car_width = element['size'] * 2
                car_height = element['size']
                
                pygame.draw.rect(
                    road_surface,
                    element['color'],
                    (int(element_x - car_width/2), int(element_y - car_height/2), car_width, car_height),
                    0,
                    3
                )
                
                pygame.draw.rect(
                    road_surface,
                    (200, 200, 255),
                    (int(element_x - car_width/3), int(element_y - car_height/3), car_width/3, car_height/3),
                    0,
                    2
                )
                
                wheel_size = element['size'] / 4
                pygame.draw.circle(
                    road_surface,
                    (0, 0, 0),
                    (int(element_x - car_width/3), int(element_y + car_height/2)),
                    int(wheel_size)
                )
                pygame.draw.circle(
                    road_surface,
                    (0, 0, 0),
                    (int(element_x + car_width/3), int(element_y + car_height/2)),
                    int(wheel_size)
                )
                
            elif element['type'] == 'building':
                pygame.draw.rect(
                    road_surface,
                    element['color'],
                    (int(element_x - element['width']/2), int(element_y - element['height']/2), 
                     element['width'], element['height'])
                )
                
                window_size = min(8, element['width'] / 6)
                num_floors = int(element['height'] / (window_size * 2))
                num_columns = int(element['width'] / (window_size * 2))
                
                for floor in range(num_floors):
                    for col in range(num_columns):
                        window_x = element_x - element['width']/2 + window_size + col * window_size * 2
                        window_y = element_y - element['height']/2 + window_size + floor * window_size * 2
                        
                        if random.random() < 0.3:
                            window_color = (255, 255, 0)
                        else:
                            window_color = (100, 100, 150)
                            
                        pygame.draw.rect(
                            road_surface,
                            window_color,
                            (int(window_x), int(window_y), int(window_size), int(window_size))
                        )
                        
            elif element['type'] == 'billboard':
                pygame.draw.rect(
                    road_surface,
                    element['color'],
                    (int(element_x - element['width']/2), int(element_y - element['height']/2), 
                     element['width'], element['height'])
                )
                
                pygame.draw.rect(
                    road_surface,
                    (0, 0, 0),
                    (int(element_x - element['width']/2), int(element_y - element['height']/2), 
                     element['width'], element['height']),
                    2
                )
                
                billboard_font = pygame.font.SysFont(None, min(24, int(element['width'] / len(element['text']))))
                billboard_text = billboard_font.render(element['text'], True, (0, 0, 0))
                billboard_rect = billboard_text.get_rect(center=(int(element_x), int(element_y)))
                road_surface.blit(billboard_text, billboard_rect)
                
                pygame.draw.rect(
                    road_surface,
                    (100, 100, 100),
                    (int(element_x - element['width']/4), int(element_y + element['height']/2), 
                     4, element['height'] * 1.5)
                )
                pygame.draw.rect(
                    road_surface,
                    (100, 100, 100),
                    (int(element_x + element['width']/4 - 4), int(element_y + element['height']/2), 
                     4, element['height'] * 1.5)
                )
        
        # Draw animated road markings with distinct visibility
        lanes = 3  # Number of lanes on the road
        lane_width = self.road_width / lanes
        
        # Draw chevron patterns on the road for motion effect
        chevron_spacing = 100
        chevron_offset = int(self.scroll_y * 2) % chevron_spacing  # Make chevrons move at 2x speed
        
        for i in range(-5, 30):  # Create enough chevrons to fill the screen
            y = center_y - chevron_offset + i * chevron_spacing
            
            for lane_idx in range(1, lanes):
                lane_x = center_x - self.road_width / 2 + lane_idx * lane_width
                
                # Draw a chevron (V shape pointing up)
                chevron_width = lane_width * 0.4
                chevron_height = 30
                
                points = [
                    (lane_x, y - chevron_height/2),  # Top point
                    (lane_x - chevron_width/2, y + chevron_height/2),  # Bottom left
                    (lane_x + chevron_width/2, y + chevron_height/2)   # Bottom right
                ]
                
                # Use alternating colors for better visibility
                chevron_color = (220, 220, 0) if i % 2 == 0 else (200, 200, 200)
                
                pygame.draw.polygon(road_surface, chevron_color, points, 0)
        
        # Draw track markings (enhanced visibility)
        for marking in self.track_markings:
            base_x = center_x + (marking['x'] - self.screen_width / 2)
            base_y = center_y + (marking['y'] - self.screen_height / 2)
            
            # Calculate screen position with offset, using the same logic as road_elements for consistency
            current_marking_offset_x = element_world_offset_x % road_surface.get_width() # Using element_world_offset for markings too
            current_marking_offset_y = element_world_offset_y % road_surface.get_height()

            movement_multiplier_marking = 1.2 # Can be different if needed
            element_x = (base_x - current_marking_offset_x * movement_multiplier_marking) % road_surface.get_width()
            element_y = (base_y - current_marking_offset_y * movement_multiplier_marking) % road_surface.get_height()
            
            # Draw different marker styles for left/right sides
            if marking['side'] == 'left':
                # Left side: white post with red reflector
                pygame.draw.rect(
                    road_surface,
                    (255, 255, 255),
                    (int(element_x - 2), int(element_y - 20), 4, 40)
                )
                pygame.draw.circle(
                    road_surface,
                    (255, 0, 0),
                    (int(element_x), int(element_y - 15)),
                    4
                )
            else:
                # Right side: white post with green reflector
                pygame.draw.rect(
                    road_surface,
                    (255, 255, 255),
                    (int(element_x - 2), int(element_y - 20), 4, 40)
                )
                pygame.draw.circle(
                    road_surface,
                    (0, 255, 0),
                    (int(element_x), int(element_y - 15)),
                    4
                )
            
            # Draw number on every 10th marker for mile markers
            if marking['number'] > 0 and marking['number'] % 1 == 0:
                font_size = 16
                marker_font = pygame.font.SysFont(None, font_size)
                number_text = marker_font.render(str(marking['number']), True, (0, 0, 0))
                
                # Position the text on a white background square
                bg_size = max(18, number_text.get_width() + 8)
                pygame.draw.rect(
                    road_surface,
                    (255, 255, 255),
                    (int(element_x - bg_size/2), int(element_y + 2), bg_size, bg_size)
                )
                
                number_rect = number_text.get_rect(
                    center=(int(element_x), int(element_y + 2 + bg_size/2))
                )
                road_surface.blit(number_text, number_rect)
        
        # Draw particles for dynamic motion effect
        for particle in self.particles:
            # Convert world coordinates to screen coordinates for particles
            # Particles are drawn relative to the road_surface, their x,y are already world-like
            # but need to be mapped onto the potentially larger road_surface and account for its own scrolling.
            # The particle positions are updated in self.update based on car movement (scroll_x, scroll_y logic).
            # So, their positions are already "world" positions.
            # To draw them on the road_surface, they need to be offset by the same amount as other elements.
            
            particle_on_surface_x = (particle['x'] - element_world_offset_x * particle['speed']) % road_surface.get_width()
            particle_on_surface_y = (particle['y'] - element_world_offset_y * particle['speed']) % road_surface.get_height()


            # Make particles fade in/out based on speed
            if hasattr(self, 'speed'):
                alpha = int(min(255, self.speed * 500))
                if alpha > 0:
                    # Using elongated particles for higher speeds adds to motion effect
                    if self.speed > 0.3:
                        elongation = min(5, int(self.speed * 10))
                        pygame.draw.line(
                            road_surface,
                            (*particle['color'], min(255, alpha)),
                            (int(particle_on_surface_x), int(particle_on_surface_y)),
                            (int(particle_on_surface_x), int(particle_on_surface_y + elongation)),
                            particle['size']
                        )
                    else:
                        # Regular circle for slower speeds
                        pygame.draw.circle(
                            road_surface,
                            particle['color'], # Alpha already applied if needed
                            (int(particle_on_surface_x), int(particle_on_surface_y)),
                            particle['size']
                        )

        # ציור חלקיקי מהירות (אפקט מהירות)
        for particle in self.speed_particles:
            alpha = int(120 + 120 * min(1.0, self.speed))
            color = (255, 255, 255, alpha)
            s = pygame.Surface((particle['size'], 12), pygame.SRCALPHA)
            pygame.draw.rect(s, color, (0, 0, particle['size'], 12))
            screen.blit(s, (int(particle['x']), int(particle['y'])))
        
        # סיבוב משטח המסלול כולו
        rotated_road = pygame.transform.rotate(road_surface, self.rotation)
        rotated_rect = rotated_road.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        screen.blit(rotated_road, rotated_rect)
    
    def reset(self):
        """Reset the road to initial state"""
        self.scroll_y = 0
        self.scroll_x = 0
        self.rotation = 0
        self._generate_road_elements(50)  # Generate new random elements
        self._generate_track_markings(100)  # Generate new track markings
        self._generate_particles(50)  # Generate new particles