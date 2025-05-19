#!/usr/bin/env python
"""
Game Launcher for Hand Gesture Car Control System
Connects hand gestures to car controls
"""

import os
import sys
import time
import pygame
import cv2
import numpy as np
import random
import math
from hand_detector.connection import HandCarConnectionManager
from game.car import Car
from game.obstacle import ObstacleManager, Obstacle

class GameLauncher:
    """Main game launcher class"""
    
    def __init__(self):
        """Initialize the game launcher"""
        pygame.init()
        pygame.display.set_caption("Hand Gesture Car Control")
        
        # Set up the screen
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width + 320, self.screen_height))  # Add width for camera feed
        
        # World dimensions (bigger than screen)
        self.world_width = 6000  # Much larger world
        self.world_height = 6000  # Much larger world
        
        # World offset (for camera that follows car)
        self.world_offset_x = 0
        self.world_offset_y = 0
        
        # Track generation settings
        self.track_segments = []
        self.last_segment_end = (self.world_width // 2, self.world_height // 2)
        self.track_width = 300
        self.segment_length = 500
        self.segments_ahead = 10  # How many segments to generate ahead
        self.segments_total = 0  # Counter for total segments generated
        
        # Set up the clock
        self.clock = pygame.time.Clock()
        self.target_fps = 60
        
        # Create the car at center of world
        self.car = Car(self.world_width // 2, self.world_height // 2)
        self.car.set_screen_dimensions(self.screen_width, self.screen_height)
        self.car.set_world_dimensions(self.world_width, self.world_height)
        
        # Create obstacle manager and generate initial obstacles
        self.obstacle_manager = ObstacleManager()
        
        # Generate initial track
        self._generate_track_segment(self.segments_ahead)
        
        # Create the hand-car connection
        self.connection = HandCarConnectionManager()
        
        # Game state
        self.running = True
        self.debug_mode = True
        self.show_help = True
        self.show_camera = True  # Always show camera feed
        self.last_obstacle_check = time.time()
        
        # Camera display
        self.camera_surface = None
        
        # Grid properties for ground
        self.grid_size = int(100)  # וודא שזה שלם
        self.grid_color = (180, 180, 180)
        self.background_color = (240, 240, 240)
        
        # Generate some random ground elements
        self.ground_elements = []
        for _ in range(100):
            x = random.randint(0, self.world_width)
            y = random.randint(0, self.world_height)
            size = random.randint(5, 15)
            color_value = random.randint(160, 220)
            self.ground_elements.append({
                'x': x,
                'y': y,
                'size': size,
                'color': (color_value, color_value, color_value)
            })
        
        # Fonts
        self.font = pygame.font.SysFont(None, 24)
        self.title_font = pygame.font.SysFont(None, 36)
        
        # Create a separate OpenCV window for the camera feed
        self.show_opencv_window = True
        self.opencv_window_name = "Hand Detector Camera"
        if self.show_opencv_window:
            cv2.namedWindow(self.opencv_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.opencv_window_name, 640, 480)
        
        # Game timer settings
        self.game_duration = 3 * 60  # 3 minutes in seconds
        self.start_time = None
        self.elapsed_time = 0
        self.time_remaining = self.game_duration
        self.game_completed = False
        
        # Score tracking
        self.score = 0
        self.distance_traveled = 0
        self.last_position = None
        
        print("🎮 Game launcher initialized")
        
    def _generate_track_segment(self, num_segments=1):
        """Generate new track segments ahead of the car - primarily northward direction"""
        for _ in range(num_segments):
            # Determine the starting point of the new segment
            start_x, start_y = self.last_segment_end
            
            # קביעת כיוון צפון (0 מעלות) עם סטייה קטנה בלבד
            if self.track_segments:
                # הגבלת הסטייה המקסימלית ל-±7 מעלות בכל מקטע כדי ליצור מסלול יותר צפוני
                new_direction = 0 + random.uniform(-7, 7)
                
                # וידוא שהמסלול לא יסטה יותר מ-45 מעלות מצפון אחרי סדרת פניות
                if len(self.track_segments) > 10:
                    # ככל שהמסלול מתארך, מגבירים את הנטייה לחזור לכיוון צפון
                    correction_factor = 0.2  # גורם תיקון של 20%
                    new_direction = new_direction * (1 - correction_factor)  # התקרבות הדרגתית לכיוון 0 (צפון)
            else:
                # מקטע ראשון - תמיד צפונה (0 מעלות)
                new_direction = 0
            
            # הגדלת אורך המקטע כדי שהמסלול יהיה ארוך יותר
            # ככל שהזמן למשחק קצר יותר, המקטעים מתארכים יותר
            if hasattr(self, 'elapsed_time') and hasattr(self, 'game_duration'):
                # חישוב מקדם הגדלה שגדל ככל שהמשחק מתקדם - למנוע סיום המסלול
                progress_factor = min(2.5, 1.0 + (self.elapsed_time / self.game_duration))
                segment_length = self.segment_length * 2.0 * progress_factor
            else:
                segment_length = self.segment_length * 2.0
                
            # Calculate end point based on direction and segment length
            direction_rad = math.radians(new_direction)
            end_x = start_x + math.sin(direction_rad) * segment_length
            end_y = start_y - math.cos(direction_rad) * segment_length  # הערה: -cos כי ציר Y הפוך בפיתגון
            
            # חישוב בגבולות העולם - הגדלת העולם אם צריך ולא רק הגבלה
            # זה יוודא שהמסלול לא "ייתקע" בקצה העולם
            boundary_padding = self.track_width * 2
            
            # בדיקה אם המקטע הבא יוצא מגבולות העולם הנוכחי
            if end_x < boundary_padding or end_x > self.world_width - boundary_padding:
                # במקום להגביל, נכפה כיוון צפון מוחלט
                new_direction = 0
                # חישוב מחדש של נקודת הסיום
                direction_rad = math.radians(new_direction)
                end_x = start_x + math.sin(direction_rad) * segment_length
                end_y = start_y - math.cos(direction_rad) * segment_length
            
            # יצירת מקטע מסלול חדש
            new_segment = {
                "start": (start_x, start_y),
                "end": (end_x, end_y),
                "direction": new_direction,
                "width": self.track_width,
                "obstacles": []
            }
            
            # הוספת המקטע לרשימת המקטעים
            self.track_segments.append(new_segment)
            self.last_segment_end = (end_x, end_y)
            self.segments_total += 1
            
            # Generate obstacles along the side of the track
            self._generate_obstacles_for_segment(new_segment)

    def _generate_obstacles_for_segment(self, segment):
        """Generate obstacles for a track segment"""
        start_x, start_y = segment["start"]
        end_x, end_y = segment["end"]
        direction = segment["direction"]
        width = segment["width"]
        
        # Calculate perpendicular direction (for obstacles on both sides)
        perp_direction = direction + 90
        perp_direction_rad = math.radians(perp_direction)
        
        # Number of obstacles to generate on each side
        num_obstacles = random.randint(1, 3)
        
        # Generate obstacles on both sides of the track
        for side in [-1, 1]:  # -1 for left side, 1 for right side
            for i in range(num_obstacles):
                # Position along the segment (0.0 to 1.0)
                t = random.uniform(0.1, 0.9)
                
                # Calculate position on the track
                pos_x = start_x + (end_x - start_x) * t
                pos_y = start_y + (end_y - start_y) * t
                
                # Calculate offset perpendicular to the track
                offset_distance = width * 0.7 + random.uniform(30, 100)  # Outside track but not too far
                offset_x = math.sin(perp_direction_rad) * offset_distance * side
                offset_y = -math.cos(perp_direction_rad) * offset_distance * side
                
                # Final obstacle position
                obs_x = pos_x + offset_x
                obs_y = pos_y + offset_y
                
                # Ensure obstacle is within world bounds
                if (0 < obs_x < self.world_width and 0 < obs_y < self.world_height):
                    # Choose random obstacle type
                    obs_type = random.choice(["rock", "tree", "cone", "puddle"])
                    size = random.randint(30, 60)
                    
                    # Create obstacle
                    obstacle = Obstacle(obs_x, obs_y, obs_type, size, size)
                    
                    # Add to obstacles list for this segment
                    segment["obstacles"].append(obstacle)
                    
                    # Add to main obstacle manager
                    self.obstacle_manager.obstacles.append(obstacle)
    
    def _check_track_generation(self):
        """בדיקה אם יש צורך לייצר מקטעי מסלול חדשים - עם שיפורים למניעת סיום המסלול"""
        if self.track_segments:
            # חישוב מרחק מסוף המסלול הנוכחי
            last_end_x, last_end_y = self.last_segment_end
            distance_to_last = math.sqrt((self.car.x - last_end_x)**2 + (self.car.y - last_end_y)**2)
            
            # יצירת מקטעים חדשים אם המכונית מתקרבת לסוף המסלול
            # הגדלת מרחק הסף להתחיל לייצר מוקדם יותר
            distance_threshold = self.segment_length * 8  # הגדלת הסף מ-5 ל-8
            
            if distance_to_last < distance_threshold:
                # הגדלת מספר המקטעים החדשים שנוצרים בכל פעם
                segments_to_create = 8  # הגדלה מ-5 ל-8
                
                # תוספת משמעותית יותר של מקטעים כשמתקרבים לסוף המשחק
                if hasattr(self, 'time_remaining') and self.time_remaining < 60:  # פחות מדקה
                    segments_to_create = 15  # יצירת הרבה יותר מקטעים בסוף המשחק
                
                self._generate_track_segment(segments_to_create)
                print(f"יצירת {segments_to_create} מקטעי מסלול חדשים. סה\"כ: {len(self.track_segments)}")
                return True
            
            # שיפור מינימום המקטעים הנדרשים
            min_segments_ahead = 25  # הגדלה מ-15 ל-25
            
            # כשמתקרבים לסוף המשחק, נדרשים הרבה יותר מקטעים
            if hasattr(self, 'time_remaining'):
                if self.time_remaining < 60:  # פחות מדקה
                    min_segments_ahead = 50
                elif self.time_remaining < 30:  # פחות מחצי דקה
                    min_segments_ahead = 80
            
            if len(self.track_segments) < min_segments_ahead:
                segments_to_add = min_segments_ahead - len(self.track_segments)
                self._generate_track_segment(segments_to_add)
                print(f"יצירת {segments_to_add} מקטעי מסלול לשמירה על מינימום. סה\"כ: {len(self.track_segments)}")
                return True
        
        return False
    
    def _draw_track(self, screen, offset_x, offset_y):
        """Draw the track segments"""
        for segment in self.track_segments:
            # Convert world coordinates to screen coordinates
            start_x, start_y = segment["start"]
            end_x, end_y = segment["end"]
            width = segment["width"]
            
            screen_start_x = start_x - offset_x + self.screen_width // 2
            screen_start_y = start_y - offset_y + self.screen_height // 2
            screen_end_x = end_x - offset_x + self.screen_width // 2
            screen_end_y = end_y - offset_y + self.screen_height // 2
            
            # Only draw segments that are visible on screen (with padding)
            padding = width * 2
            if (screen_start_x > -padding and screen_start_x < self.screen_width + padding and
                screen_start_y > -padding and screen_start_y < self.screen_height + padding) or \
               (screen_end_x > -padding and screen_end_x < self.screen_width + padding and
                screen_end_y > -padding and screen_end_y < self.screen_height + padding):
                
                # Calculate direction vector
                dx = end_x - start_x
                dy = end_y - start_y
                length = math.sqrt(dx*dx + dy*dy)
                
                if length > 0:
                    # Normalize
                    dx /= length
                    dy /= length
                    
                    # Calculate perpendicular vector
                    perpx = -dy
                    perpy = dx
                    
                    # Calculate four corners of the segment (as a rectangle)
                    half_width = width / 2
                    points = [
                        (screen_start_x + perpx * half_width, screen_start_y + perpy * half_width),
                        (screen_end_x + perpx * half_width, screen_end_y + perpy * half_width),
                        (screen_end_x - perpx * half_width, screen_end_y - perpy * half_width),
                        (screen_start_x - perpx * half_width, screen_start_y - perpy * half_width)
                    ]
                    
                    # Draw track segment as polygon
                    pygame.draw.polygon(screen, (100, 100, 100), points)  # Gray track
                    
                    # Draw track borders
                    pygame.draw.line(screen, (255, 255, 255), 
                                     (screen_start_x + perpx * half_width, screen_start_y + perpy * half_width),
                                     (screen_end_x + perpx * half_width, screen_end_y + perpy * half_width), 3)
                    pygame.draw.line(screen, (255, 255, 255),
                                     (screen_start_x - perpx * half_width, screen_start_y - perpy * half_width),
                                     (screen_end_x - perpx * half_width, screen_end_y - perpy * half_width), 3)
    
    def run(self):
        """Run the game loop"""
        # Start the hand-car connection
        if not self.connection.start():
            print("❌ Failed to start hand-car connection")
            # Display a message to the user
            self.show_error_message("Failed to initialize camera",
                                     "The game will continue without hand gesture controls.")
            return
            
        print("🏁 Starting game loop")
        
        # Initialize start time
        self.start_time = time.time()
        self.last_position = (self.car.x, self.car.y)
        
        # Main game loop
        last_time = time.time()
        track_generation_timer = 0  # טיימר לאכיפת יצירת מקטעים גם ללא התקדמות
        
        while self.running:
            # Calculate delta time
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # Update game timer
            self.elapsed_time = current_time - self.start_time
            self.time_remaining = max(0, self.game_duration - self.elapsed_time)
            
            # Check if game should end after 3 minutes
            if self.elapsed_time >= self.game_duration and not self.game_completed:
                self.game_completed = True
                print(f"⏱️ Game time (3 minutes) elapsed! Final score: {int(self.score)}")
                
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        # If game is completed, exit, otherwise confirm
                        if self.game_completed or self._confirm_exit():
                            self.running = False
                    elif event.key == pygame.K_h:
                        self.show_help = not self.show_help
                    elif event.key == pygame.K_d:
                        self.debug_mode = not self.debug_mode
                    elif event.key == pygame.K_c:
                        self.show_camera = not self.show_camera
                    elif event.key == pygame.K_v:
                        self.show_opencv_window = not self.show_opencv_window
                        if self.show_opencv_window:
                            cv2.namedWindow(self.opencv_window_name, cv2.WINDOW_NORMAL)
                            cv2.resizeWindow(self.opencv_window_name, 640, 480)
                        else:
                            cv2.destroyWindow(self.opencv_window_name)
            
            # Get controls from hand detector
            controls = self.connection.get_controls()
            
            # Get camera feed and data panel
            try:
                camera_frame, data_panel, fps = self.connection.get_visuals()
                
                # Convert camera frame to pygame surface if available
                if camera_frame is not None:
                    # Display the camera feed in OpenCV window
                    if self.show_opencv_window:
                        try:
                            cv2.imshow(self.opencv_window_name, camera_frame)
                            cv2.waitKey(1)
                        except Exception as e:
                            print(f"❌ Error showing OpenCV window: {e}")
                    
                    # Convert OpenCV frame to Pygame surface for in-game display
                    try:
                        # Resize the frame for in-game display
                        small_frame = cv2.resize(camera_frame, (320, 240))
                        # Convert BGR to RGB for Pygame
                        small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                        # Create Pygame surface
                        self.camera_surface = pygame.surfarray.make_surface(small_frame_rgb.swapaxes(0, 1))
                    except Exception as e:
                        print(f"❌ Error converting camera frame: {e}")
                        self.camera_surface = None
            except Exception as e:
                print(f"❌ Error getting camera visuals: {e}")
            
            # Update car if game is not completed
            if not self.game_completed:
                self.car.update(controls, dt)
                
                # עדכון משופר של יצירת המסלול - מריץ בכל פריים
                new_segments_created = self._check_track_generation()
                
                # מנגנון נוסף: יצירת מקטעים חדשים כל X שניות, ללא תלות במיקום השחקן
                # זה מבטיח שהמסלול ימשיך להתפתח גם אם השחקן נעצר במקום
                track_generation_timer += dt
                if track_generation_timer > 5.0:  # כל 5 שניות
                    track_generation_timer = 0
                    # יצירת מקטעים נוספים באופן יזום
                    segments_to_add = max(3, int(15 * (1 - self.time_remaining / self.game_duration)))
                    self._generate_track_segment(segments_to_add)
                    print(f"יצירה תקופתית: {segments_to_add} מקטעי מסלול נוספים. סה\"כ: {len(self.track_segments)}")
                
                # בדיקה נוספת לקראת סוף המשחק (שיפור של הקוד הקיים)
                if self.elapsed_time >= self.game_duration * 0.8:  # ב-80% מזמן המשחק
                    # וידוא שיש מספיק מקטעים לסיום
                    if len(self.track_segments) < 100:  # מספר גבוה מאוד בסוף המשחק
                        additional_segments = 100 - len(self.track_segments)
                        self._generate_track_segment(additional_segments)
                        print(f"קרוב לסיום המשחק! יצירת {additional_segments} מקטעי מסלול. סה\"כ: {len(self.track_segments)}")
                
                # Calculate distance traveled (for score)
                current_pos = (self.car.x, self.car.y)
                distance = math.sqrt((current_pos[0] - self.last_position[0])**2 + 
                                    (current_pos[1] - self.last_position[1])**2)
                self.distance_traveled += distance
                self.last_position = current_pos
                
                # Add to score based on distance and speed
                self.score += distance * 0.01 * (1 + self.car.speed)
                
                # Bonus score for using boost
                if controls.get('boost', False):
                    self.score += dt * 5  # Bonus points for boosting
                
                # Update world offset to center on car
                self.world_offset_x = self.car.x - self.screen_width // 2
                self.world_offset_y = self.car.y - self.screen_height // 2
                
                # Update obstacles
                self.obstacle_manager.update(dt)
                
                # Check for collisions with obstacles
                if current_time - self.last_obstacle_check > 0.1:
                    self.last_obstacle_check = current_time
                    collisions = self.obstacle_manager.check_collisions(self.car)
                    for obstacle in collisions:
                        self.car.handle_obstacle_collision(obstacle.type)
                        # Reduce score on collision
                        self.score = max(0, self.score - 10)
            
            # Draw everything
            self._draw()
            
            # Cap the frame rate
            self.clock.tick(self.target_fps)
            
        # Clean up
        self.connection.stop()
        cv2.destroyAllWindows()
        pygame.quit()
    
    def show_error_message(self, title, message):
        """Show an error message to the user"""
        screen = self.screen
        screen.fill((0, 0, 0))
        
        # Create fonts
        title_font = pygame.font.SysFont(None, 48)
        message_font = pygame.font.SysFont(None, 32)
        instruction_font = pygame.font.SysFont(None, 24)
        
        # Render text
        title_surface = title_font.render(title, True, (255, 50, 50))
        message_surface = message_font.render(message, True, (255, 255, 255))
        instruction_surface = instruction_font.render("Press any key to continue", True, (200, 200, 200))
        
        # Position text
        title_rect = title_surface.get_rect(center=(self.screen_width//2, self.screen_height//2 - 60))
        message_rect = message_surface.get_rect(center=(self.screen_width//2, self.screen_height//2))
        instruction_rect = instruction_surface.get_rect(center=(self.screen_width//2, self.screen_height//2 + 60))
        
        # Draw text
        screen.blit(title_surface, title_rect)
        screen.blit(message_surface, message_rect)
        screen.blit(instruction_surface, instruction_rect)
        
        # Update the display
        pygame.display.flip()
        
        # Wait for key press
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    waiting = False
    
    def _confirm_exit(self):
        """Ask player for confirmation before exiting before 3 minutes"""
        if self.elapsed_time >= self.game_duration:
            return True
            
        font = pygame.font.SysFont(None, 36)
        
        # Create confirmation overlay
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))  # Semi-transparent overlay
        
        # Draw confirmation message
        text1 = font.render("Are you sure you want to exit?", True, (255, 255, 255))
        text2 = font.render("The game will end before the 3 minutes timer!", True, (255, 200, 50))
        text3 = font.render("Press ESC again to confirm, any other key to continue", True, (255, 255, 255))
        
        # Position text
        text1_rect = text1.get_rect(center=(self.screen_width//2, self.screen_height//2 - 40))
        text2_rect = text2.get_rect(center=(self.screen_width//2, self.screen_height//2))
        text3_rect = text3.get_rect(center=(self.screen_width//2, self.screen_height//2 + 40))
        
        # Display confirmation
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text1, text1_rect)
        self.screen.blit(text2, text2_rect)
        self.screen.blit(text3, text3_rect)
        pygame.display.flip()
        
        # Wait for key press
        waiting = True
        confirmed = False
        while waiting and self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                    self.running = False  # Quitting window means exit
                    confirmed = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        waiting = False
                        confirmed = True
                    else:
                        waiting = False
                        
        return confirmed
    
    def _draw(self):
        """Draw the game screen"""
        try:
            # Clear the screen
            self.screen.fill(self.background_color)
            
            # Draw the grid and ground elements
            self._draw_world()
            
            # Draw the track
            self._draw_track(self.screen, self.world_offset_x, self.world_offset_y)
            
            # Draw obstacles
            self.obstacle_manager.draw(self.screen, self.world_offset_x, self.world_offset_y)
            
            # Draw the car in center of screen
            self.car.draw(self.screen, self.world_offset_x, self.world_offset_y)
            
            # Draw camera feed if available and enabled
            if self.camera_surface is not None and self.show_camera:
                # Draw the camera feed on the right side
                self.screen.blit(self.camera_surface, (self.screen_width, 0))
                
                # Add label
                camera_label = self.font.render("Camera Feed (Press C to toggle)", True, (0, 0, 0))
                self.screen.blit(camera_label, (self.screen_width + 10, 245))
            else:
                # Draw a placeholder if camera feed is not available
                camera_rect = pygame.Rect(self.screen_width, 0, 320, 240)
                pygame.draw.rect(self.screen, (30, 30, 30), camera_rect)
                
                # Draw a message
                no_cam_font = pygame.font.SysFont(None, 30)
                no_cam_text = no_cam_font.render("Camera Not Available", True, (255, 50, 50))
                self.screen.blit(no_cam_text, (self.screen_width + 60, 110))
            
            # Draw timer and score
            self._draw_hud()
            
            # Draw game over message if completed
            if self.game_completed:
                self._draw_game_completed()
            
            # Draw debug info if enabled
            if self.debug_mode:
                controls = self.connection.get_controls()
                debug_text = [
                    f"FPS: {self.clock.get_fps():.1f}",
                    f"Gesture: {controls.get('gesture_name', 'Unknown')}",
                    f"Steering: {controls.get('steering', 0):.2f}",
                    f"Throttle: {controls.get('throttle', 0):.2f}",
                    f"Braking: {controls.get('braking', False)}",
                    f"Boost: {controls.get('boost', False)}",
                    f"World position: ({self.car.x:.1f}, {self.car.y:.1f})",
                    f"Rotation: {self.car.rotation:.1f}°",
                    f"Speed: {self.car.speed:.2f}",
                    f"Health: {self.car.health}",
                    f"Track segments: {len(self.track_segments)}"
                ]
                
                for i, text in enumerate(debug_text):
                    text_surface = self.font.render(text, True, (0, 0, 0))
                    self.screen.blit(text_surface, (10, 10 + i * 25))
                
            # Draw help text if enabled
            if self.show_help:
                help_text = [
                    "Controls:",
                    "- Move hand left/right: Steer",
                    "- Raise/lower hand: Speed up/down",
                    "- Make a fist: Brake",
                    "- Thumb up: Boost",
                    "- Open palm: Stop",
                    "",
                    "Keys:",
                    "- ESC: Quit",
                    "- H: Toggle this help",
                    "- D: Toggle debug info",
                    "- C: Toggle camera feed in game",
                    "- V: Toggle separate camera window"
                ]
                
                # Draw help panel on right side, below camera
                if self.show_camera:
                    help_y = 280
                else:
                    help_y = 10
                    
                help_surface = pygame.Surface((320, 300), pygame.SRCALPHA)
                pygame.draw.rect(help_surface, (255, 255, 255, 200), (0, 0, 320, 300), 0)
                pygame.draw.rect(help_surface, (0, 0, 0), (0, 0, 320, 300), 2)
                
                title_surface = self.title_font.render("Help", True, (0, 0, 0))
                help_surface.blit(title_surface, (10, 10))
                
                for i, text in enumerate(help_text):
                    text_surface = self.font.render(text, True, (0, 0, 0))
                    help_surface.blit(text_surface, (10, 50 + i * 20))
                    
                self.screen.blit(help_surface, (self.screen_width, help_y))
                
            # Update the display
            pygame.display.flip()
            
        except Exception as e:
            print(f"Error in draw: {e}")
            import traceback
            traceback.print_exc()
    
    def _draw_hud(self):
        """Draw heads-up display with timer and score"""
        # Create semi-transparent background for timer
        timer_bg = pygame.Surface((300, 50), pygame.SRCALPHA)
        timer_bg.fill((0, 0, 0, 128))  # Semi-transparent black
        
        # Convert remaining time to minutes:seconds format
        minutes = int(self.time_remaining // 60)
        seconds = int(self.time_remaining % 60)
        
        # Choose timer color based on remaining time
        if self.time_remaining > 60:  # More than 1 minute
            timer_color = (255, 255, 255)  # White
        elif self.time_remaining > 30:  # 30-60 seconds
            timer_color = (255, 255, 0)  # Yellow
        elif self.time_remaining > 10:  # 10-30 seconds
            timer_color = (255, 165, 0)  # Orange
        else:  # Less than 10 seconds
            timer_color = (255, 0, 0)  # Red
            # Make it flash during the last 10 seconds
            if int(self.time_remaining * 2) % 2 == 0:
                timer_color = (255, 255, 255)
        
        # Format timer text
        time_text = f"Time: {minutes:02d}:{seconds:02d}"
        score_text = f"Score: {int(self.score)}"
        
        # Create font objects
        timer_font = pygame.font.SysFont(None, 36)
        timer_render = timer_font.render(time_text, True, timer_color)
        score_render = timer_font.render(score_text, True, (255, 255, 255))
        
        # Position and draw timer
        self.screen.blit(timer_bg, (10, 10))
        self.screen.blit(timer_render, (20, 20))
        self.screen.blit(score_render, (170, 20))
        
        # Draw progress bar under timer
        progress_width = 280
        progress_height = 8
        bar_filled = max(0, min(1, 1 - (self.time_remaining / self.game_duration))) * progress_width
        
        # Background of progress bar
        pygame.draw.rect(self.screen, (100, 100, 100), (20, 45, progress_width, progress_height))
        # Filled portion of progress bar
        pygame.draw.rect(self.screen, timer_color, (20, 45, int(bar_filled), progress_height))
        
    def _draw_game_completed(self):
        """Draw game completed/over message"""
        # Create semi-transparent overlay
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))  # Semi-transparent black
        
        # Create font objects
        big_font = pygame.font.SysFont(None, 72)
        medium_font = pygame.font.SysFont(None, 48)
        small_font = pygame.font.SysFont(None, 36)
        
        # Game over text
        game_over_text = big_font.render("Time's Up!", True, (255, 255, 255))
        final_score_text = medium_font.render(f"Final Score: {int(self.score)}", True, (255, 215, 0))
        
        # Instructions
        instruction_text = small_font.render("Press ESC to exit", True, (200, 200, 200))
        
        # Calculate positions
        game_over_rect = game_over_text.get_rect(center=(self.screen_width//2, self.screen_height//2 - 60))
        score_rect = final_score_text.get_rect(center=(self.screen_width//2, self.screen_height//2))
        instruction_rect = instruction_text.get_rect(center=(self.screen_width//2, self.screen_height//2 + 70))
        
        # Draw overlay and text
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(game_over_text, game_over_rect)
        self.screen.blit(final_score_text, score_rect)
        self.screen.blit(instruction_text, instruction_rect)
        
    def _draw_world(self):
        """
        מצייר את העולם (רקע, רשת, אלמנטים נוספים)
        """
        # חישוב הגבולות של החלק הנראה מהעולם
        start_x = max(0, int(self.world_offset_x // self.grid_size * self.grid_size))
        end_x = min(self.world_width, int((self.world_offset_x + self.screen_width) // self.grid_size * self.grid_size + self.grid_size * 2))
        
        start_y = max(0, int(self.world_offset_y // self.grid_size * self.grid_size))
        end_y = min(self.world_height, int((self.world_offset_y + self.screen_height) // self.grid_size * self.grid_size + self.grid_size * 2))
        
        # ציור קווי רשת אופקיים - המרה ל-int להימנע משגיאה
        for y in range(int(start_y), int(end_y), int(self.grid_size)):
            screen_y = int(y - self.world_offset_y)
            pygame.draw.line(
                self.screen,
                self.grid_color,
                (0, screen_y),
                (self.screen_width, screen_y)
            )
        
        # ציור קווי רשת אנכיים - המרה ל-int להימנע משגיאה
        for x in range(int(start_x), int(end_x), int(self.grid_size)):
            screen_x = int(x - self.world_offset_x)
            pygame.draw.line(
                self.screen,
                self.grid_color,
                (screen_x, 0),
                (screen_x, self.screen_height)
            )
        
        # ציור אלמנטי קרקע
        for element in self.ground_elements:
            screen_x = int(element['x'] - self.world_offset_x)
            screen_y = int(element['y'] - self.world_offset_y)
            
            # ציור רק אם האלמנט נראה על המסך
            if (0 <= screen_x <= self.screen_width and 
                0 <= screen_y <= self.screen_height):
                pygame.draw.circle(
                    self.screen,
                    element['color'],
                    (screen_x, screen_y),
                    element['size']
                )
        
        # ציור גבולות העולם
        if self.world_offset_x < 0:
            pygame.draw.rect(
                self.screen,
                (100, 100, 100),
                (0, 0, -self.world_offset_x, self.screen_height)
            )
        
        if self.world_offset_x + self.screen_width > self.world_width:
            pygame.draw.rect(
                self.screen,
                (100, 100, 100),
                (self.world_width - self.world_offset_x, 0, 
                 self.screen_width - (self.world_width - self.world_offset_x), 
                 self.screen_height)
            )
        
        if self.world_offset_y < 0:
            pygame.draw.rect(
                self.screen,
                (100, 100, 100),
                (0, 0, self.screen_width, -self.world_offset_y)
            )
        
        if self.world_offset_y + self.screen_height > self.world_height:
            pygame.draw.rect(
                self.screen,
                (100, 100, 100),
                (0, self.world_height - self.world_offset_y,
                 self.screen_width,
                 self.screen_height - (self.world_height - self.world_offset_y))
            )
            
    def cleanup(self):
        """Clean up resources"""
        try:
            self.connection.stop()
        except:
            pass
            
        try:
            cv2.destroyAllWindows()
        except:
            pass
            
        try:
            pygame.quit()
        except:
            pass
            
        print("🧹 Game resources cleaned up")