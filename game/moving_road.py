"""
Moving Road Module for Hand Gesture Car Control Game - SYNCHRONIZED VERSION

This module implements a moving road background that responds to car movement,
creating the illusion of forward movement with FIXED DIRECTION animation.
"""

import pygame
import math
import random
import numpy as np

class MovingRoadGenerator:
    """SYNCHRONIZED Moving Road Generator with Fixed Direction Animation"""
    
    def __init__(self, screen_width, screen_height):
        """
        Initialize the moving road generator - SYNCHRONIZED VERSION
        
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
        
        # Scrolling state - FIXED FOR CORRECT DIRECTION
        self.scroll_y = 0
        self.scroll_x = 0
        self.rotation = 0
        
        # Scroll speed parameters
        self.max_scroll_speed = 500
        self.car_max_speed = 300
        
        # Current speed for effects
        self.speed = 0.0
        
        # Initialize road elements
        self.road_elements = []
        self.track_markings = []
        self.particles = []
        self.speed_particles = []
        
        # Generate all road elements
        self._generate_road_elements(50)
        self._generate_track_markings(100)
        self._generate_particles(50)
        self._generate_speed_particles(30)
        
        print("✅ MovingRoadGenerator initialized - SYNCHRONIZED VERSION")
    
    def _generate_road_elements(self, count):
        """Generate road side elements with enhanced variety"""
        self.road_elements = []
        
        # Basic elements (trees, rocks, bushes)
        for _ in range(count):
            side = random.choice([-1, 1])
            x_offset = side * (self.road_width / 2 + random.randint(20, 100))
            y = random.randint(-500, self.screen_height + 2000)
            
            element_type = random.choice(['rock', 'bush', 'tree', 'pole'])
            
            if element_type == 'rock':
                size = random.randint(8, 18)
                color = (120, 120, 120)
            elif element_type == 'bush':
                size = random.randint(12, 22)
                color = (0, 120, 0)
            elif element_type == 'tree':
                size = random.randint(18, 30)
                color = (0, 100, 0)
            else:  # pole
                size = random.randint(6, 12)
                color = (200, 200, 200)
            
            self.road_elements.append({
                'x': self.screen_width / 2 + x_offset,
                'y': y,
                'size': size,
                'color': color,
                'type': element_type
            })
        
        # Enhanced road infrastructure
        self._generate_infrastructure()
        
        print(f"✅ Generated {len(self.road_elements)} road elements")
    
    def _generate_infrastructure(self):
        """Generate road infrastructure elements"""
        # Distance markers
        marker_spacing = 200
        for i in range(20):
            y_pos = -500 + i * marker_spacing
            
            # Left side markers
            self.road_elements.append({
                'x': self.screen_width / 2 - self.road_width / 2 - 30,
                'y': y_pos,
                'size': 8,
                'color': (255, 255, 255),
                'type': 'distance_marker',
                'side': 'left',
                'number': i
            })
            
            # Right side markers
            self.road_elements.append({
                'x': self.screen_width / 2 + self.road_width / 2 + 30,
                'y': y_pos,
                'size': 8,
                'color': (255, 255, 255),
                'type': 'distance_marker',
                'side': 'right',
                'number': i
            })
        
        # Traffic signs
        sign_colors = [(255, 0, 0), (255, 255, 0), (0, 255, 0)]
        for i in range(8):
            side = random.choice([-1, 1])
            self.road_elements.append({
                'x': self.screen_width / 2 + side * (self.road_width / 2 + 70),
                'y': -300 + i * 600 + random.randint(-50, 50),
                'size': 15,
                'color': random.choice(sign_colors),
                'type': 'traffic_sign',
                'text': random.choice(['STOP', 'SLOW', 'GO'])
            })
        
        # Buildings in background
        building_colors = [(100, 100, 100), (120, 100, 80), (80, 80, 100)]
        for i in range(12):
            side = random.choice([-1, 1])
            distance = random.randint(150, 400)
            
            self.road_elements.append({
                'x': self.screen_width / 2 + side * (self.road_width / 2 + distance),
                'y': -800 + i * 500 + random.randint(-100, 100),
                'width': random.randint(60, 120),
                'height': random.randint(80, 150),
                'color': random.choice(building_colors),
                'type': 'building'
            })
    
    def _generate_track_markings(self, count):
        """Generate track markings for motion emphasis"""
        self.track_markings = []
        
        for i in range(count):
            y_pos = -1000 + i * 80
            
            # Alternating left/right markers
            side = 'left' if i % 2 == 0 else 'right'
            x_pos = (self.screen_width / 2 - self.road_width / 2 - 15 if side == 'left' 
                    else self.screen_width / 2 + self.road_width / 2 + 15)
            
            self.track_markings.append({
                'x': x_pos,
                'y': y_pos,
                'size': 6,
                'type': 'marker',
                'number': i // 10,
                'side': side
            })
    
    def _generate_particles(self, count):
        """Generate particles for motion effects"""
        self.particles = []
        
        for _ in range(count):
            self.particles.append({
                'x': random.randint(0, self.screen_width),
                'y': random.randint(-1000, self.screen_height + 1000),
                'size': random.randint(1, 3),
                'speed': random.uniform(0.5, 2.0),
                'color': (220, 220, 220)
            })
    
    def _generate_speed_particles(self, count):
        """Generate speed particles for high-speed effects"""
        self.speed_particles = []
        
        for _ in range(count):
            self.speed_particles.append({
                'x': random.randint(0, self.screen_width),
                'y': random.randint(0, self.screen_height),
                'speed': random.uniform(2.0, 4.0),
                'size': random.randint(1, 3),
                'trail_length': random.randint(5, 15)
            })

    def update(self, rotation, speed, dt):
        """
        Update the moving road state - FIXED DIRECTION FOR FORWARD MOVEMENT
        
        Args:
            rotation: Car rotation in degrees
            speed: Normalized car speed (0.0 to 1.0)
            dt: Time delta in seconds
        """
        # Store current speed for effects
        self.speed = speed
        
        # Calculate scroll speed
        base_scroll_speed = self.max_scroll_speed * speed
        angle_rad = math.radians(rotation)
        
        # Update scroll values - FIXED FOR FORWARD MOVEMENT
        # For forward movement, road elements should move DOWN and TOWARDS the car
        self.scroll_x += base_scroll_speed * math.sin(angle_rad) * dt
        self.scroll_y += base_scroll_speed * math.cos(angle_rad) * dt  # Positive = forward
        
        # Update particles - FIXED DIRECTION
        for particle in self.particles:
            # Particles move down for forward movement
            particle['y'] += base_scroll_speed * particle['speed'] * dt
            particle['x'] += random.uniform(-10, 10) * dt  # Small random movement
            
            # Wrap around screen
            if particle['y'] > self.screen_height + 100:
                particle['y'] = -100
                particle['x'] = random.randint(0, self.screen_width)
            
            if particle['x'] < -50 or particle['x'] > self.screen_width + 50:
                particle['x'] = random.randint(0, self.screen_width)
        
        # Update speed particles - FIXED DIRECTION
        for particle in self.speed_particles:
            # Speed particles move down faster for forward movement
            particle['y'] += base_scroll_speed * particle['speed'] * dt
            
            # Reset when off screen
            if particle['y'] > self.screen_height + 20:
                particle['y'] = -20
                particle['x'] = random.randint(0, self.screen_width)

    def draw(self, screen, world_offset_x=0, world_offset_y=0):
        """
        Draw the moving road - SYNCHRONIZED VERSION with FIXED DIRECTION
        
        Args:
            screen: Pygame surface to draw on
            world_offset_x: World offset X (optional)
            world_offset_y: World offset Y (optional)
        """
        # Fill background with grass
        screen.fill(self.grass_color)
        
        # Draw main road surface
        self._draw_road_surface(screen)
        
        # Draw lane markings with animation
        self._draw_lane_markings(screen)
        
        # Draw road elements
        self._draw_road_elements(screen, world_offset_x, world_offset_y)
        
        # Draw particles and effects
        self._draw_particles(screen)
        
        # Draw speed effects if moving fast
        if self.speed > 0.3:
            self._draw_speed_effects(screen)
    
    def _draw_road_surface(self, screen):
        """Draw the basic road surface"""
        road_rect = pygame.Rect(
            self.screen_width // 2 - self.road_width // 2,
            0,
            self.road_width,
            self.screen_height
        )
        pygame.draw.rect(screen, self.road_color, road_rect)
    
    def _draw_lane_markings(self, screen):
        """Draw animated lane markings - FIXED DIRECTION"""
        # Center line - animated dashes
        center_x = self.screen_width // 2
        
        # FIXED: For forward movement, use positive scroll_y
        dash_cycle = self.lane_length + self.lane_gap
        y_offset = self.scroll_y % dash_cycle
        
        y = -y_offset
        while y < self.screen_height + dash_cycle:
            if y % dash_cycle < self.lane_length:
                dash_start = max(0, y)
                dash_end = min(self.screen_height, y + self.lane_length)
                
                if dash_end > dash_start:
                    pygame.draw.rect(screen, self.lane_color,
                                   (center_x - self.lane_width // 2, dash_start,
                                    self.lane_width, dash_end - dash_start))
            y += dash_cycle
        
        # Side lane markings
        for lane in [1, 2]:  # Two additional lanes
            lane_x = (self.screen_width // 2 - self.road_width // 2 + 
                     self.road_width * lane // 3)
            
            y = -y_offset
            while y < self.screen_height + dash_cycle:
                if y % dash_cycle < self.lane_length:
                    dash_start = max(0, y)
                    dash_end = min(self.screen_height, y + self.lane_length)
                    
                    if dash_end > dash_start:
                        pygame.draw.rect(screen, self.lane_color,
                                       (lane_x - self.lane_width // 2, dash_start,
                                        self.lane_width, dash_end - dash_start))
                y += dash_cycle
    
    def _draw_road_elements(self, screen, world_offset_x, world_offset_y):
        """Draw road side elements with parallax"""
        for element in self.road_elements:
            # Calculate position with scroll offset
            element_x = element['x'] - self.scroll_x * 0.8
            element_y = element['y'] - self.scroll_y * 0.8
            
            # Only draw if visible
            if (-100 < element_x < self.screen_width + 100 and
                -100 < element_y < self.screen_height + 100):
                
                self._draw_single_element(screen, element, element_x, element_y)
    
    def _draw_single_element(self, screen, element, x, y):
        """Draw a single road element"""
        if element['type'] == 'rock':
            pygame.draw.circle(screen, element['color'], (int(x), int(y)), element['size'])
            
        elif element['type'] == 'bush':
            pygame.draw.circle(screen, element['color'], (int(x), int(y)), element['size'])
            # Add texture
            pygame.draw.circle(screen, (0, 150, 0), (int(x), int(y)), element['size'] - 3)
            
        elif element['type'] == 'tree':
            # Draw trunk
            trunk_width = element['size'] // 4
            trunk_height = element['size'] * 2
            pygame.draw.rect(screen, (101, 67, 33),
                           (int(x - trunk_width // 2), int(y - trunk_height // 2),
                            trunk_width, trunk_height))
            
            # Draw foliage
            pygame.draw.circle(screen, element['color'], (int(x), int(y - trunk_height // 2)), 
                             element['size'])
            
        elif element['type'] == 'pole':
            pygame.draw.circle(screen, element['color'], (int(x), int(y)), element['size'])
            
        elif element['type'] == 'distance_marker':
            # Draw marker post
            marker_height = element['size'] * 6
            pygame.draw.rect(screen, element['color'],
                           (int(x - 2), int(y - marker_height // 2), 4, marker_height))
            
            # Draw reflector
            reflector_color = (255, 0, 0) if element['side'] == 'left' else (0, 255, 0)
            pygame.draw.circle(screen, reflector_color, (int(x), int(y - marker_height // 4)), 4)
            
        elif element['type'] == 'traffic_sign':
            # Draw sign background
            pygame.draw.circle(screen, element['color'], (int(x), int(y)), element['size'])
            
            # Draw sign text
            font = pygame.font.SysFont(None, max(12, element['size']))
            text = font.render(element['text'], True, (0, 0, 0))
            text_rect = text.get_rect(center=(int(x), int(y)))
            screen.blit(text, text_rect)
            
        elif element['type'] == 'building':
            # Draw building
            pygame.draw.rect(screen, element['color'],
                           (int(x - element['width'] // 2), int(y - element['height'] // 2),
                            element['width'], element['height']))
            
            # Add windows
            window_size = 6
            for floor in range(0, element['height'], window_size * 3):
                for col in range(0, element['width'], window_size * 2):
                    window_x = int(x - element['width'] // 2 + col + window_size // 2)
                    window_y = int(y - element['height'] // 2 + floor + window_size // 2)
                    
                    # Random lit windows
                    window_color = (255, 255, 0) if random.random() < 0.3 else (100, 100, 150)
                    pygame.draw.rect(screen, window_color,
                                   (window_x, window_y, window_size, window_size))
    
    def _draw_particles(self, screen):
        """Draw motion particles"""
        for particle in self.particles:
            if 0 <= particle['x'] <= self.screen_width and 0 <= particle['y'] <= self.screen_height:
                # Draw particle with alpha based on speed
                alpha = int(100 + 155 * min(1.0, self.speed))
                
                if self.speed > 0.4:
                    # Draw elongated particles for high speed
                    trail_length = int(self.speed * 20)
                    pygame.draw.line(screen, particle['color'],
                                   (int(particle['x']), int(particle['y'])),
                                   (int(particle['x']), int(particle['y'] - trail_length)),
                                   particle['size'])
                else:
                    # Draw regular particles
                    pygame.draw.circle(screen, particle['color'],
                                     (int(particle['x']), int(particle['y'])), 
                                     particle['size'])
    
    def _draw_speed_effects(self, screen):
        """Draw speed-based visual effects"""
        for particle in self.speed_particles:
            if 0 <= particle['x'] <= self.screen_width and 0 <= particle['y'] <= self.screen_height:
                # Speed lines that get longer with higher speed
                trail_length = int(particle['trail_length'] * self.speed * 2)
                
                # Create speed line surface with alpha
                alpha = int(60 + 195 * min(1.0, self.speed))
                line_surface = pygame.Surface((particle['size'], trail_length), pygame.SRCALPHA)
                
                # Gradient effect
                for i in range(trail_length):
                    line_alpha = int(alpha * (1 - i / trail_length))
                    color = (255, 255, 255, line_alpha)
                    pygame.draw.rect(line_surface, color, (0, i, particle['size'], 1))
                
                screen.blit(line_surface, (int(particle['x']), int(particle['y'])))
        
        # Screen edge blur effect at high speeds
        if self.speed > 0.7:
            self._draw_speed_blur(screen)
    
    def _draw_speed_blur(self, screen):
        """Draw speed blur effect around screen edges"""
        blur_intensity = int((self.speed - 0.7) * 100)
        
        # Create blur surfaces
        blur_width = int(50 * (self.speed - 0.7))
        
        # Left and right blur
        for side in ['left', 'right']:
            blur_surface = pygame.Surface((blur_width, self.screen_height), pygame.SRCALPHA)
            
            for x in range(blur_width):
                alpha = int(blur_intensity * (1 - x / blur_width))
                color = (255, 255, 255, alpha)
                pygame.draw.line(blur_surface, color, (x, 0), (x, self.screen_height))
            
            if side == 'left':
                screen.blit(blur_surface, (0, 0))
            else:
                screen.blit(blur_surface, (self.screen_width - blur_width, 0))
    
    def draw_chevrons(self, screen):
        """Draw chevron patterns for enhanced motion effect"""
        if self.speed < 0.2:
            return
            
        chevron_spacing = 80
        chevron_offset = int(self.scroll_y * 3) % chevron_spacing
        
        center_x = self.screen_width // 2
        
        for i in range(-2, int(self.screen_height / chevron_spacing) + 2):
            y = chevron_offset + i * chevron_spacing
            
            # Draw chevron (arrow pointing down for forward movement)
            points = [
                (center_x, y + 15),  # Bottom point
                (center_x - 20, y - 15),  # Top left
                (center_x + 20, y - 15)   # Top right
            ]
            
            # Color based on speed
            intensity = int(100 + 155 * min(1.0, self.speed))
            color = (intensity, intensity, 0)  # Yellow
            
            pygame.draw.polygon(screen, color, points, 0)
            pygame.draw.polygon(screen, (255, 255, 255), points, 2)  # White outline
    
    def reset(self):
        """Reset the road to initial state"""
        self.scroll_y = 0
        self.scroll_x = 0
        self.rotation = 0
        self.speed = 0.0
        
        # Regenerate elements for variety
        self._generate_road_elements(50)
        self._generate_track_markings(100)
        self._generate_particles(50)
        self._generate_speed_particles(30)
        
        print("✅ MovingRoad reset - SYNCHRONIZED VERSION")
    
    def get_road_bounds(self):
        """Get road boundaries for collision detection"""
        left_bound = self.screen_width // 2 - self.road_width // 2
        right_bound = self.screen_width // 2 + self.road_width // 2
        
        return {
            'left': left_bound,
            'right': right_bound,
            'top': 0,
            'bottom': self.screen_height,
            'center': self.screen_width // 2,
            'width': self.road_width
        }
    
    def add_dynamic_element(self, element_type, x, y, **kwargs):
        """Add a dynamic element to the road (for future expansion)"""
        element = {
            'x': x,
            'y': y,
            'type': element_type,
            'dynamic': True,
            **kwargs
        }
        
        # Set default properties based on type
        if element_type == 'obstacle':
            element.setdefault('size', 20)
            element.setdefault('color', (255, 0, 0))
        elif element_type == 'powerup':
            element.setdefault('size', 15)
            element.setdefault('color', (0, 255, 0))
        
        self.road_elements.append(element)
        return element
    
    def remove_dynamic_elements(self):
        """Remove all dynamic elements (obstacles, powerups, etc.)"""
        self.road_elements = [elem for elem in self.road_elements 
                            if not elem.get('dynamic', False)]
    
    def get_debug_info(self):
        """Get debug information about the road state"""
        return {
            'scroll_x': self.scroll_x,
            'scroll_y': self.scroll_y,
            'speed': self.speed,
            'rotation': self.rotation,
            'elements_count': len(self.road_elements),
            'particles_count': len(self.particles),
            'speed_particles_count': len(self.speed_particles),
            'version': 'SYNCHRONIZED'
        }

# Utility functions for road generation
def create_curved_road_section(start_x, start_y, end_x, end_y, curve_intensity=50):
    """Create a curved road section (for future use)"""
    points = []
    steps = 20
    
    for i in range(steps + 1):
        t = i / steps
        
        # Quadratic bezier curve
        mid_x = (start_x + end_x) / 2 + curve_intensity
        mid_y = (start_y + end_y) / 2
        
        x = (1-t)**2 * start_x + 2*(1-t)*t * mid_x + t**2 * end_x
        y = (1-t)**2 * start_y + 2*(1-t)*t * mid_y + t**2 * end_y
        
        points.append((x, y))
    
    return points

def generate_road_texture(width, height, road_color=(80, 80, 80)):
    """Generate a textured road surface (for future use)"""
    surface = pygame.Surface((width, height))
    surface.fill(road_color)
    
    # Add texture noise
    for _ in range(width * height // 100):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        
        # Slight color variation
        variation = random.randint(-10, 10)
        color = tuple(max(0, min(255, c + variation)) for c in road_color)
        surface.set_at((x, y), color)
    
    return surface

# Export main class
__all__ = ['MovingRoadGenerator', 'create_curved_road_section', 'generate_road_texture']

if __name__ == "__main__":
    # Test the moving road generator
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("MovingRoad Test - SYNCHRONIZED VERSION")
    clock = pygame.time.Clock()
    
    road = MovingRoadGenerator(800, 600)
    running = True
    speed = 0.0
    
    print("🎮 Testing MovingRoad - SYNCHRONIZED VERSION")
    print("Use UP/DOWN arrows to control speed")
    
    while running:
        dt = clock.tick(60) / 1000.0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Handle input
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            speed = min(1.0, speed + dt)
        elif keys[pygame.K_DOWN]:
            speed = max(0.0, speed - dt)
        else:
            speed *= 0.95  # Gradual slowdown
        
        # Update and draw
        road.update(0, speed, dt)
        road.draw(screen)
        
        # Draw debug info
        font = pygame.font.Font(None, 36)
        speed_text = font.render(f"Speed: {speed:.2f}", True, (255, 255, 255))
        screen.blit(speed_text, (10, 10))
        
        debug_info = road.get_debug_info()
        info_text = font.render(f"Scroll Y: {debug_info['scroll_y']:.1f}", True, (255, 255, 255))
        screen.blit(info_text, (10, 50))
        
        pygame.display.flip()
    
    pygame.quit()
    print("👋 MovingRoad test ended")