"""
Moving Road Module for Hand Gesture Car Control Game

This module implements a moving road background that responds to car movement,
creating the illusion that the car is moving while actually staying in the center
of the screen.
"""

import pygame
import math
import random

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
        
        # Speed tracking for particles and grid
        self.speed = 0
        
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
        
        # Add colorful objects to visualize movement
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
        # Save speed for use in drawing
        self.speed = speed
        
        # Calculate scroll speed based on car speed
        base_scroll_speed = self.max_scroll_speed * speed
        
        # Convert rotation to radians
        angle_rad = math.radians(rotation)
        
        # Update scroll values based on car movement
        self.scroll_x += base_scroll_speed * math.sin(angle_rad) * dt
        self.scroll_y += base_scroll_speed * math.cos(angle_rad) * dt
        
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

    def draw(self, screen, world_offset_x=0, world_offset_y=0):
        """
        Draw the moving road on screen
        
        Args:
            screen: Pygame surface for drawing
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
        
        # Road elements can still use world_offset_x/y for parallax if needed
        element_world_offset_x = int(self.scroll_x + world_offset_x * 1.5)
        element_world_offset_y = int(self.scroll_y + world_offset_y * 1.5)
        
        # ציור אלמנטי המסלול
        element_offset_x = int(self.scroll_x) % road_surface.get_width()
        element_offset_y = int(self.scroll_y) % road_surface.get_height()
        
        for element in self.road_elements:
            # המרת קואורדינטות למערכת הקואורדינטות של משטח המסלול
            base_x = center_x + (element['x'] - self.screen_width / 2)
            base_y = center_y + (element['y'] - self.screen_height / 2)
            
            # החלת היסטי גלילה
            element_x = (base_x - element_offset_x) % road_surface.get_width()
            element_y = (base_y - element_offset_y) % road_surface.get_height()
            
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
            
            # Calculate screen position with offset
            element_x = (base_x - element_offset_x) % road_surface.get_width()
            element_y = (base_y - element_offset_y) % road_surface.get_height()
            
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
        
        # Draw particles for dynamic motion effect
        for particle in self.particles:
            # Use particle position directly (already updated in update method)
            particle_screen_x = int(particle['x'])
            particle_screen_y = int(particle['y'])
            
            # Make particles more visible based on speed
            if self.speed > 0.1:
                # Using elongated particles for higher speeds adds to motion effect
                if self.speed > 0.3:
                    elongation = min(5, int(self.speed * 10))
                    pygame.draw.line(
                        road_surface,
                        particle['color'],
                        (particle_screen_x, particle_screen_y),
                        (particle_screen_x, particle_screen_y + elongation),
                        particle['size']
                    )
                else:
                    # Regular circle for slower speeds
                    pygame.draw.circle(
                        road_surface,
                        particle['color'],
                        (particle_screen_x, particle_screen_y),
                        particle['size']
                    )

        # סיבוב משטח המסלול כולו
        rotated_road = pygame.transform.rotate(road_surface, self.rotation)
        
        # מיקום משטח המסלול המסובב במרכז המסך
        rotated_rect = rotated_road.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        
        # ציור המסלול המסובב על המסך
        screen.blit(rotated_road, rotated_rect)
    
    def reset(self):
        """Reset the road to initial state"""
        self.scroll_y = 0
        self.scroll_x = 0
        self.rotation = 0
        self._generate_road_elements(50)  # Generate new random elements
        self._generate_track_markings(100)  # Generate new track markings
        self._generate_particles(50)  # Generate new particles