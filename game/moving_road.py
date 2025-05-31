"""
Moving Road Module for Hand Gesture Car Control Game - SIMPLE VERSION

This module implements a simple moving road background with only lane markings.
"""

import pygame
import math

class MovingRoadGenerator:
    """Simple Moving Road Generator - Only lane markings"""
    
    def __init__(self, screen_width, screen_height):
        """
        Initialize the moving road generator - SIMPLE VERSION
        
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
        self.speed = 0.0
        
        print("✅ MovingRoadGenerator initialized - SIMPLE VERSION")
    
    def update(self, rotation, speed, dt):
        """
        Update the moving road state - SIMPLE VERSION
        
        Args:
            rotation: Car rotation in degrees (ignored)
            speed: Normalized car speed (0.0 to 1.0)
            dt: Time delta in seconds
        """
        # Store current speed
        self.speed = speed
        
        # Calculate scroll speed
        base_scroll_speed = 500 * speed
        
        # Update scroll position
        self.scroll_y += base_scroll_speed * dt

    def draw(self, screen, world_offset_x=0, world_offset_y=0):
        """
        Draw the simple moving road
        
        Args:
            screen: Pygame surface to draw on
            world_offset_x: World offset X (ignored)
            world_offset_y: World offset Y (ignored)
        """
        # Fill background with grass
        screen.fill(self.grass_color)
        
        # Draw main road surface
        road_rect = pygame.Rect(
            self.screen_width // 2 - self.road_width // 2,
            0,
            self.road_width,
            self.screen_height
        )
        pygame.draw.rect(screen, self.road_color, road_rect)
        
        # Draw moving center line
        self._draw_center_line(screen)
    
    def _draw_center_line(self, screen):
        """Draw simple moving center line - FIXED to match orange cones direction"""
        center_x = self.screen_width // 2
        line_width = 6
        
        # Moving dashed line - REVERSED direction to match cones
        dash_spacing = self.lane_length + self.lane_gap
        offset = int(-self.scroll_y) % dash_spacing  # NEGATIVE to reverse direction
        
        y = -offset
        while y < self.screen_height + dash_spacing:
            if y + self.lane_length > 0 and y < self.screen_height:
                dash_start = max(0, y)
                dash_end = min(self.screen_height, y + self.lane_length)
                
                pygame.draw.rect(screen, self.lane_color,
                               (center_x - line_width // 2, dash_start,
                                line_width, dash_end - dash_start))
            
            y += dash_spacing
    
    def reset(self):
        """Reset the road to initial state"""
        self.scroll_y = 0
        self.speed = 0.0
        
        print("✅ MovingRoad reset - SIMPLE VERSION")
    
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

# Export main class
__all__ = ['MovingRoadGenerator']

if __name__ == "__main__":
    # Test the moving road generator
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("MovingRoad Test - SIMPLE VERSION")
    clock = pygame.time.Clock()
    
    road = MovingRoadGenerator(800, 600)
    running = True
    speed = 0.0
    
    print("🎮 Testing MovingRoad - SIMPLE VERSION")
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
        info_text = font.render(f"All Objects Fixed: {debug_info['version']}", True, (0, 255, 0))
        screen.blit(info_text, (10, 50))
        
        pygame.display.flip()
    
    pygame.quit()
    print("👋 SIMPLE test ended")
