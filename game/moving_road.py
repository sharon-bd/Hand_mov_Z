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
        self.scroll_x = 0
        self.speed = 0.0
        
        # Add steering offset support
        self.steering_offset = 0
        
        print("✅ MovingRoadGenerator initialized - SIMPLE VERSION")
    
    def set_steering_offset(self, offset):
        """Set steering offset for consistent road movement"""
        self.steering_offset = offset
    
    def update(self, rotation, speed, dt):
        """
        Update the moving road state - WITH STRONGER STEERING OFFSET
        
        Args:
            rotation: Car rotation in degrees
            speed: Normalized car speed (0.0 to 1.0)
            dt: Time delta in seconds
        """
        # Store current speed
        self.speed = speed
        
        # Calculate scroll speed
        base_scroll_speed = 500 * speed
        
        # Update vertical scroll position
        self.scroll_y += base_scroll_speed * dt
        
        # ENHANCED: Stronger horizontal scroll based on car rotation
        # Convert rotation to steering effect with more sensitivity
        steering_factor = rotation / 30.0  # 30 degrees = full steering (more sensitive)
        steering_factor = max(-1.5, min(1.5, steering_factor))  # Allow more extreme values
        
        # Move background horizontally opposite to car direction with stronger effect
        horizontal_speed = steering_factor * 150 * speed  # Increased from 100 to 150
        self.scroll_x += horizontal_speed * dt

    def draw(self, screen, world_offset_x=0, world_offset_y=0):
        """
        Draw the simple moving road with horizontal offset compensation
        
        Args:
            screen: Pygame surface to draw on
            world_offset_x: Additional world offset X
            world_offset_y: Additional world offset Y
        """
        # Fill background with grass
        screen.fill(self.grass_color)
        
        # FIXED: Calculate road position with steering offset compensation
        # Road moves opposite to steering offset to stay centered
        road_center_x = self.screen_width // 2 - int(self.scroll_x) - int(self.steering_offset)
        
        # Draw main road surface with offset
        road_rect = pygame.Rect(
            road_center_x - self.road_width // 2,
            0,
            self.road_width,
            self.screen_height
        )
        pygame.draw.rect(screen, self.road_color, road_rect)
        
        # Draw moving center line with offset
        self._draw_center_line(screen, road_center_x)
    
    def _draw_center_line(self, screen, center_x):
        """Draw simple moving center line with horizontal offset compensation"""
        line_width = 6
        
        # Moving dashed line - REVERSED direction to match cones
        dash_spacing = self.lane_length + self.lane_gap
        offset = int(-self.scroll_y) % dash_spacing  # NEGATIVE to reverse direction
        
        y = -offset
        while y < self.screen_height + dash_spacing:
            if y + self.lane_length > 0 and y < self.screen_height:
                dash_start = max(0, y)
                dash_end = min(self.screen_height, y + self.lane_length)
                
                # Only draw if line is within screen bounds with steering compensation
                if center_x - line_width // 2 >= -50 and center_x + line_width // 2 <= self.screen_width + 50:
                    pygame.draw.rect(screen, self.lane_color,
                                   (center_x - line_width // 2, dash_start,
                                    line_width, dash_end - dash_start))
            
            y += dash_spacing
    
    def get_debug_info(self):
        """Get debug information for testing"""
        return {
            'version': 'Simple Road 1.0',
            'scroll_y': self.scroll_y,
            'scroll_x': self.scroll_x,
            'speed': self.speed
        }
    
    def reset(self):
        """Reset the road to initial state"""
        self.scroll_y = 0
        self.scroll_x = 0
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
