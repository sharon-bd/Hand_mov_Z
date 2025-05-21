#!/usr/bin/env python
"""
Start Game Module for Hand Gesture Car Control Game

This module implements the main game logic and serves as the entry point
for the specific game modes.
"""

import pygame
import sys
import time
import random
import math
import os

# Import other game modules
from .car import Car
from .physics import PhysicsEngine
from .objects import ObstacleManager, PowerUpManager, ScoreManager
from .renderer import GameRenderer
from .audio_manager import AudioManager
from .camera_manager import CameraManager

class Game:
    """Main Game class that manages the overall game state"""
    
    def __init__(self, screen_width=800, screen_height=600, hand_detector=None):
        """Initialize the game launcher"""
        pygame.init()
        pygame.display.set_caption("Hand Gesture Car Control")
        
        # Set up the screen
        self.screen_width = screen_width
        self.screen_height = screen_height
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
        from hand_detector.connection import HandCarConnectionManager
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
            import cv2
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
        
        # יצירת מחלקת הרנדור
        self.renderer = GameRenderer(self.screen_width, self.screen_height)
        
        print("🎮 Game launcher initialized")
    
    def set_mode(self, mode, config=None):
        """
        Set the game mode
        
        Args:
            mode: Game mode string ("easy", "normal", etc.)
            config: Optional mode configuration
        """
        self.current_mode = mode
        
        # Set default difficulty parameters
        if mode == "practice":
            # Practice mode (no obstacles)
            self.obstacle_manager.spawn_rate = 0.0
            self.obstacle_manager.obstacle_speed = 150
            self.score_manager.set_multiplier(0.5)
            self.time_limit = None
        elif mode == "easy":
            # Easy mode
            self.obstacle_manager.spawn_rate = 0.01
            self.obstacle_manager.obstacle_speed = 150
            self.score_manager.set_multiplier(1.0)
            self.time_limit = None
        elif mode == "normal":
            # Normal mode
            self.obstacle_manager.spawn_rate = 0.02
            self.obstacle_manager.obstacle_speed = 200
            self.score_manager.set_multiplier(1.5)
            self.time_limit = None
        elif mode == "hard":
            # Hard mode
            self.obstacle_manager.spawn_rate = 0.03
            self.obstacle_manager.obstacle_speed = 250
            self.score_manager.set_multiplier(2.0)
            self.time_limit = None
        elif mode == "time_trial":
            # Time trial mode
            self.obstacle_manager.spawn_rate = 0.015
            self.obstacle_manager.obstacle_speed = 225
            self.score_manager.set_multiplier(2.5)
            self.time_limit = 120  # 2 minutes
        
        # Apply custom config if provided
        if config:
            if 'spawn_rate' in config:
                self.obstacle_manager.spawn_rate = config['spawn_rate']
            if 'obstacle_speed' in config:
                self.obstacle_manager.obstacle_speed = config['obstacle_speed']
            if 'score_multiplier' in config:
                self.score_manager.set_multiplier(config['score_multiplier'])
            if 'time_limit' in config:
                self.time_limit = config['time_limit']
    
    def start(self):
        """Start the game"""
        # Reset the game state
        self.running = True
        self.paused = False
        self.game_over = False
        self.start_time = time.time()
        self.elapsed_time = 0
        
        # Reset score
        self.score_manager.reset_score()
        
        # Reset car
        self.car.x = self.screen_width // 2
        self.car.y = self.screen_height - 100
        self.car.direction = 0.0
        self.car.speed = 0.0
        self.car.health = 100
        
        # Clear obstacles and power-ups
        self.obstacle_manager.clear_obstacles()
        self.power_up_manager.power_ups = []
        
        # Start the game loop
        self.run()
    
    def run(self):
        """לולאת המשחק הראשית"""
        # אתחול חיבור היד-מכונית
        if not self.connection.start():
            print("❌ נכשל באתחול חיבור היד-מכונית")
            self.show_error_message("לא ניתן לאתחל את המצלמה",
                                  "המשחק ימשיך ללא שליטה במחוות ידיים.")
            return
        
        print("🏁 מתחיל לולאת משחק")
        
        # וידוא שהמסך מאותחל כראוי
        if not pygame.display.get_surface():
            print("⚠️ אין משטח תצוגה - יוצר מחדש")
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width + 320, self.screen_height))
        
        # אתחול זמנים
        self.start_time = time.time()
        self.last_position = (self.car.x, self.car.y)
        last_time = time.time()
        
        while self.running:
            # חישוב זמן דלתא
            current_time = time.time()
            dt = min(current_time - last_time, 0.1)  # הגבלת dt ל-0.1 שניות למניעת קפיצות גדולות
            last_time = current_time
            
            # עדכון טיימר
            self.elapsed_time = current_time - self.start_time
            self.time_remaining = max(0, self.game_duration - self.elapsed_time)
            
            # בדיקת סיום המשחק
            if self.elapsed_time >= self.game_duration and not self.game_completed:
                self.game_completed = True
                print(f"⏱️ זמן המשחק (3 דקות) הסתיים! ניקוד סופי: {int(self.score)}")
            
            # עיבוד אירועים
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
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
                            import cv2
                            cv2.namedWindow(self.opencv_window_name, cv2.WINDOW_NORMAL)
                            cv2.resizeWindow(self.opencv_window_name, 640, 480)
                        else:
                            import cv2
                            cv2.destroyWindow(self.opencv_window_name)
                    elif event.key == pygame.K_f:
                        force_segments = 20
                        self._generate_track_segment(force_segments)
                        print(f"נוצרו בכוח {force_segments} מקטעי מסלול נוספים")
            
            # קבלת פקדים ממחוות הידיים
            controls = self.connection.get_controls()
            
            # קבלת תמונת מצלמה
            try:
                import cv2
                camera_frame, data_panel, fps = self.connection.get_visuals()
                if camera_frame is not None:
                    if self.show_opencv_window:
                        try:
                            cv2.imshow(self.opencv_window_name, camera_frame)
                            cv2.waitKey(1)
                        except Exception as e:
                            print(f"❌ שגיאה בהצגת חלון OpenCV: {e}")
                    
                    # המרת תמונת המצלמה למשטח Pygame
                    try:
                        small_frame = cv2.resize(camera_frame, (320, 240))
                        small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                        self.camera_surface = pygame.surfarray.make_surface(small_frame_rgb.swapaxes(0, 1))
                    except Exception as e:
                        print(f"❌ שגיאה בהמרת תמונת המצלמה: {e}")
                        self.camera_surface = None
            except Exception as e:
                print(f"❌ שגיאה בקבלת תמונת המצלמה: {e}")
            
            # עדכון המכונית אם המשחק לא הסתיים
            if not self.game_completed:
                self.car.update(controls, dt)
                self._check_track_generation()
            
            # חישוב היסט העולם כדי לשמור על המכונית ממורכזת והדגשת התנועה
            screen_car_x = self.screen_width // 2
            screen_car_y = self.screen_height - 100  # קרוב לתחתית המסך
            
            # הגדלת פקטור ההיסט כדי להעצים את תחושת התנועה
            # גורם ההיסט גדל עם מהירות המכונית, מתחיל מ-1.0 עד 5.0
            movement_factor = 1.0 + min(4.0, self.car.speed * 5.0)
            self.world_offset_x = (self.car.x - screen_car_x) * movement_factor
            self.world_offset_y = (self.car.y - screen_car_y) * movement_factor
            
            # הוספת רעש אקראי קטן להיסט כדי ליצור תחושת תנועה נוספת
            if self.car.speed > 0.1:
                vibration = self.car.speed * 2.0  # רעידות גדלות עם המהירות
                self.world_offset_x += random.uniform(-vibration, vibration)
                self.world_offset_y += random.uniform(-vibration, vibration)
            
            # עדכון הניקוד בהתבסס על מרחק נסיעה
            current_position = (self.car.x, self.car.y)
            if self.last_position:
                distance = math.hypot(current_position[0] - self.last_position[0],
                                    current_position[1] - self.last_position[1])
                self.distance_traveled += distance
                self.score = self.distance_traveled / 10  # ניקוד פשוט בהתבסס על מרחק
            self.last_position = current_position
            
            # הכנת מצב המשחק לרנדור
            game_state = {
                'car': self.car,
                'obstacles': self.obstacle_manager.obstacles if self.obstacle_manager else [],
                'power_ups': [],  # אין כוחות כרגע
                'score': self.score,
                'health': self.car.health if hasattr(self.car, 'health') else 100,
                'time_left': self.time_remaining,
                'scroll_speed': self.car.speed * self.car.max_speed if hasattr(self.car, 'speed') and hasattr(self.car, 'max_speed') else 0,
                'dt': dt,
                'world_offset_x': self.world_offset_x,
                'world_offset_y': self.world_offset_y
            }
            
            # Pass the game state to the renderer
            self.renderer.render_game(self.screen, game_state)
            
            # Draw camera feed if available and enabled
            if self.camera_surface is not None and self.show_camera:
                self.screen.blit(self.camera_surface, (self.screen_width, 0))
                camera_label = self.font.render("Camera Feed (Press C to toggle)", True, (0, 0, 0))
                self.screen.blit(camera_label, (self.screen_width + 10, 245))
            else:
                camera_rect = pygame.Rect(self.screen_width, 0, 320, 240)
                pygame.draw.rect(self.screen, (30, 30, 30), camera_rect)
                no_cam_font = pygame.font.SysFont(None, 30)
                no_cam_text = no_cam_font.render("Camera Not Available", True, (255, 50, 50))
                self.screen.blit(no_cam_text, (self.screen_width + 60, 110))
            
            # Draw timer and score UI elements
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
                    f"Track segments: {len(self.track_segments)}",
                    f"Delta Time: {dt:.4f}s"
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
                    "- V: Toggle separate camera window",
                    "- F: Generate additional track segments"
                ]
                
                if self.show_camera:
                    help_y = 280
                else:
                    help_y = 10
                    
                help_surface = pygame.Surface((320, 320), pygame.SRCALPHA)
                pygame.draw.rect(help_surface, (255, 255, 255, 200), (0, 0, 320, 320), 0)
                pygame.draw.rect(help_surface, (0, 0, 0), (0, 0, 320, 320), 2)
                
                title_surface = self.title_font.render("Help", True, (0, 0, 0))
                help_surface.blit(title_surface, (10, 10))
                
                for i, text in enumerate(help_text):
                    text_surface = self.font.render(text, True, (0, 0, 0))
                    help_surface.blit(text_surface, (10, 50 + i * 20))
                    
                self.screen.blit(help_surface, (self.screen_width, help_y))
            
            # עדכון התצוגה
            pygame.display.flip()
            
            # הגבלת קצב הפריימים
            self.clock.tick(self.target_fps)
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    # ESC toggles pause when in game, exits when game over
                    if self.game_over:
                        self.running = False
                    else:
                        self.paused = not self.paused
                
                elif event.key == pygame.K_SPACE:
                    # SPACE advances tutorial or restarts when game over
                    if self.show_tutorial:
                        self.tutorial_step += 1
                        if self.tutorial_step >= len(self.tutorial_steps):
                            self.show_tutorial = False
                    elif self.game_over:
                        self.start()
                    elif self.paused:
                        self.paused = False
                
                # Camera controls
                elif event.key == pygame.K_c:
                    # Toggle camera display
                    self.camera.show_camera = not self.camera.show_camera
                elif event.key == pygame.K_v:
                    # Cycle to next camera
                    self.camera.cycle_camera()
    
    def update(self, dt):
        """
        Update game state
        
        Args:
            dt: Time delta in seconds
        """
        # Update elapsed time
        self.elapsed_time = time.time() - self.start_time
        
        # Check time limit if set
        if self.time_limit and self.elapsed_time >= self.time_limit:
            self.game_over = True
            return
        
        # Get hand gesture controls
        controls = self._get_hand_controls()
        
        # Update car based on controls
        self.car.update(controls, dt)
        
        # Keep car within road bounds
        self._constrain_car_to_road()
        
        # Update engine sound based on car speed
        self.audio.update_engine_sound(self.car.speed, self.car.boost_active)
        
        # Spawn and update obstacles
        if random.random() < self.obstacle_manager.spawn_rate:
            self.obstacle_manager.spawn_obstacle()
        
        obstacles = self.obstacle_manager.update(dt, self.score_manager.score)
        
        # Check for collisions with obstacles
        self._check_obstacle_collisions(obstacles)
        
        # Spawn and update power-ups
        if random.random() < self.power_up_manager.spawn_rate:
            self.power_up_manager.spawn_power_up()
            
        power_ups = self.power_up_manager.update(dt)
        
        # Check for collisions with power-ups
        self._check_power_up_collisions(power_ups)
        
        # Update score based on time
        self.score_manager.add_score(dt * 10)  # 10 points per second
    
    def draw(self):
        """Render the current game state"""
        # Create a game state dictionary to pass to the renderer
        game_state = {
            'car': self.car,
            'obstacles': self.obstacle_manager.obstacles,
            'power_ups': self.power_up_manager.power_ups,
            'score': self.score_manager.score,
            'health': self.car.health,
            'elapsed_time': self.elapsed_time,
            'time_left': self.time_limit - self.elapsed_time if self.time_limit else None,
            'scroll_speed': self.car.speed * self.car.max_speed,
            'dt': self.clock.get_time() / 1000.0
        }
        
        # Render the game
        self.renderer.render_game(self.screen, game_state)
        
        # Draw camera feed if enabled and available
        if self.camera.show_camera:
            camera_frame = self.camera.get_camera_for_display()
            if camera_frame:
                # Draw in top-right corner
                self.screen.blit(camera_frame, (self.screen_width - camera_frame.get_width() - 10, 10))
        
        # Draw additional UI
        if self.paused:
            self._draw_pause_screen()
        elif self.game_over:
            self._draw_game_over_screen()
        elif self.show_tutorial:
            self._draw_tutorial()
    
    def _draw(self):
        """ציור מסך המשחק"""
        try:
            if not pygame.display.get_surface():
                print("אין משטח תצוגה תקף! יוצר מחדש")
                self.screen = pygame.display.set_mode((self.screen_width + 320, self.screen_height))
            
            # מרכוז המכונית על המסך
            screen_car_x = self.screen_width // 2
            screen_car_y = self.screen_height - 100  # קרוב לתחתית המסך
            
            # עדכון היסט העולם כדי לשמור על המכונית ממורכזת
            self.world_offset_x = self.car.x - screen_car_x
            self.world_offset_y = self.car.y - screen_car_y
            
            # הכנת מצב המשחק לרנדור עם היסט מדויק
            game_state = {
                'car': self.car,
                'obstacles': self.obstacle_manager.obstacles if self.obstacle_manager else [],
                'power_ups': [],  # אין כוחות כרגע
                'score': self.score,
                'health': self.car.health if hasattr(self.car, 'health') else 100,
                'time_left': self.time_remaining,
                'scroll_speed': self.car.speed * self.car.max_speed if hasattr(self.car, 'speed') and hasattr(self.car, 'max_speed') else 0,
                'dt': self.clock.get_time() / 1000.0,
                'world_offset_x': self.world_offset_x,
                'world_offset_y': self.world_offset_y
            }
            
            # עדכון מיקום המכונית על המסך לרנדור
            self.car.screen_x = screen_car_x
            self.car.screen_y = screen_car_y
            
            # רנדור המשחק
            self.renderer.render_game(self.screen, game_state)
            
            # ציור תמונת המצלמה אם זמינה ומופעלת
            if self.camera_surface is not None and self.show_camera:
                self.screen.blit(self.camera_surface, (self.screen_width, 0))
                camera_label = self.font.render("תצוגת מצלמה (לחץ C לשינוי)", True, (0, 0, 0))
                self.screen.blit(camera_label, (self.screen_width + 10, 245))
            else:
                camera_rect = pygame.Rect(self.screen_width, 0, 320, 240)
                pygame.draw.rect(self.screen, (30, 30, 30), camera_rect)
                no_cam_font = pygame.font.SysFont(None, 30)
                no_cam_text = no_cam_font.render("מצלמה לא זמינה", True, (255, 50, 50))
                self.screen.blit(no_cam_text, (self.screen_width + 60, 110))
            
            # ציור HUD
            self._draw_hud()
            
            # ציור הודעת סיום משחק אם הסתיים
            if self.game_completed:
                self._draw_game_completed()
            
            # ציור מידע דיבוג אם מופעל
            if self.debug_mode:
                controls = self.connection.get_controls()
                debug_text = [
                    f"FPS: {self.clock.get_fps():.1f}",
                    f"מחווה: {controls.get('gesture_name', 'לא מזוהה')}",
                    f"היגוי: {controls.get('steering', 0):.2f}",
                    f"תאוצה: {controls.get('throttle', 0):.2f}",
                    f"בלימה: {controls.get('braking', False)}",
                    f"בוסט: {controls.get('boost', False)}",
                    f"מיקום בעולם: ({self.car.x:.1f}, {self.car.y:.1f})",
                    f"מיקום על המסך: ({screen_car_x:.1f}, {screen_car_y:.1f})",
                    f"סיבוב: {self.car.rotation:.1f}°",
                    f"מהירות: {self.car.speed:.2f}",
                    f"בריאות: {self.car.health}",
                    f"מקטעי מסלול: {len(self.track_segments)}"
                ]
                
                for i, text in enumerate(debug_text):
                    text_surface = self.font.render(text, True, (0, 0, 0))
                    self.screen.blit(text_surface, (10, 10 + i * 25))
            
            # ציור טקסט עזרה אם מופעל
            if self.show_help:
                help_text = [
                    "פקדים:",
                    "- הזז יד שמאלה/ימינה: היגוי",
                    "- הרם/הורד יד: האצה/האטה",
                    "- יד קפוצה: בלימה",
                    "- אגודל למעלה: בוסט",
                    "- כף יד פתוחה: עצירה",
                    "",
                    "מקשים:",
                    "- ESC: יציאה",
                    "- H: הפעל/כבה עזרה זו",
                    "- D: הפעל/כבה מידע דיבוג",
                    "- C: הפעל/כבה תצוגת מצלמה במשחק",
                    "- V: הפעל/כבה חלון מצלמה נפרד"
                ]
                
                help_y = 280 if self.show_camera else 10
                help_surface = pygame.Surface((320, 300), pygame.SRCALPHA)
                pygame.draw.rect(help_surface, (255, 255, 255, 200), (0, 0, 320, 300), 0)
                pygame.draw.rect(help_surface, (0, 0, 0), (0, 0, 320, 300), 2)
                
                title_surface = self.title_font.render("עזרה", True, (0, 0, 0))
                help_surface.blit(title_surface, (10, 10))
                
                for i, text in enumerate(help_text):
                    text_surface = self.font.render(text, True, (0, 0, 0))
                    help_surface.blit(text_surface, (10, 50 + i * 20))
                
                self.screen.blit(help_surface, (self.screen_width, help_y))
                
            pygame.display.update()
            print("המסך עודכן")
            
        except Exception as e:
            print(f"שגיאה קריטית בציור המסך: {e}")
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
    
    def _get_hand_controls(self):
        """
        Get control inputs from hand gestures
        
        Returns:
            Dictionary with control values
        """
        if self.hand_detector is None:
            # Return default controls if no detector
            return {
                'steering': 0.0,
                'throttle': 0.0,
                'braking': False,
                'boost': False
            }
        
        # Get the latest camera frame
        frame = self.camera.get_frame()
        
        if frame is None:
            # Return default controls if no frame
            return {
                'steering': 0.0,
                'throttle': 0.0,
                'braking': False,
                'boost': False
            }
        
        try:
            # Process the frame with the hand detector
            controls, processed_frame, data_panel = self.hand_detector.detect_gestures(frame)
            
            # Store processed frame and data panel for display
            self.camera.camera_frame = processed_frame
            
            return controls
            
        except Exception as e:
            print(f"Error processing hand gestures: {e}")
            return {
                'steering': 0.0,
                'throttle': 0.0,
                'braking': False,
                'boost': False
            }
    
    def _constrain_car_to_road(self):
        """Keep the car within the road boundaries"""
        road_left = self.road_x + self.car.width // 2
        road_right = self.road_x + self.road_width - self.car.width // 2
        
        self.car.x = max(road_left, min(road_right, self.car.x))
    
    def _check_obstacle_collisions(self, obstacles):
        """Check for collisions between car and obstacles"""
        for obstacle in obstacles:
            if not obstacle.get('hit', False) and self.car.check_collision(obstacle):
                # Mark as hit
                obstacle['hit'] = True
                
                # Apply damage to car
                damage = obstacle.get('damage', 10)
                self.car.take_damage(damage)
                
                # Play collision sound
                self.audio.play_collision_sound()
                
                # Create particle effect
                self.renderer.create_explosion(
                    obstacle['x'], 
                    obstacle['y'],
                    obstacle.get('color', (255, 0, 0)),
                    20
                )
    
    def _check_power_up_collisions(self, power_ups):
        """Check for collisions between car and power-ups"""
        for power_up in power_ups:
            if not power_up.get('collected', False):
                # Create a simple collision check (could be improved)
                dx = abs(power_up['x'] - self.car.x)
                dy = abs(power_up['y'] - self.car.y)
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance < (self.car.width + power_up['width']) / 2:
                    # Mark as collected
                    power_up['collected'] = True
                    
                    # Apply power-up effect
                    self._apply_power_up(power_up)
                    
                    # Play power-up sound
                    self.audio.play_power_up_sound()
                    
                    # Create particle effect
                    self.renderer.create_explosion(
                        power_up['x'],
                        power_up['y'],
                        power_up.get('color', (0, 255, 0)),
                        30
                    )
    
    def _apply_power_up(self, power_up):
        """Apply the effect of a power-up"""
        effect = power_up.get('effect', '')
        
        if effect == 'boost':
            # Temporary boost effect (would be handled by car logic)
            self.car.boost_active = True
            
            # Revert after duration
            def end_boost():
                self.car.boost_active = False
            
            # Schedule end of boost (this is a simple approach)
            duration = power_up.get('duration', 5.0)
            pygame.time.set_timer(pygame.USEREVENT, int(duration * 1000), 1)
            
        elif effect == 'shield':
            # Shield protection (not fully implemented)
            pass
            
        elif effect == 'repair':
            # Repair car damage
            self.car.repair(25)  # Restore 25% health
            
        # Add score for collecting power-up
        self.score_manager.add_score(50)
    
    def _check_game_over(self):
        """Check for game over conditions"""
        if self.game_over:
            return
            
        # Check car health
        if self.car.health <= 0:
            self.game_over = True
            self.audio.play_collision_sound()  # Death sound
            
        # Check time limit if set
        if self.time_limit and self.elapsed_time >= self.time_limit:
            self.game_over = True
    
    def _draw_pause_screen(self):
        """Draw the pause screen overlay"""
        # Create a semi-transparent overlay
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        # Draw pause message
        pause_text = self.font_large.render("PAUSED", True, (255, 255, 255))
        self.screen.blit(
            pause_text,
            (self.screen_width // 2 - pause_text.get_width() // 2, 200)
        )
        
        # Draw instructions
        instructions = [
            "Press SPACE to continue",
            "Press ESC to quit"
        ]
        
        y = 300
        for instruction in instructions:
            text = self.font_medium.render(instruction, True, (255, 255, 255))
            self.screen.blit(
                text,
                (self.screen_width // 2 - text.get_width() // 2, y)
            )
            y += 50
    
    def _draw_game_over_screen(self):
        """Draw the game over screen"""
        self.renderer.draw_game_over(
            self.screen,
            self.score_manager.score,
            self.score_manager.high_score
        )
    
    def _draw_tutorial(self):
        """Draw the tutorial overlay"""
        if self.tutorial_step < len(self.tutorial_steps):
            self.renderer.draw_tutorial(
                self.screen,
                self.tutorial_steps[self.tutorial_step],
                self.tutorial_step,
                len(self.tutorial_steps)
            )
    
    def cleanup(self):
        """Clean up resources before exiting"""
        self.audio.cleanup()
        self.camera.cleanup()
        pygame.quit()

    def _check_track_generation(self):
        # השבתת יצירת מקטעי מסלול מאחר ש-MovingRoadGenerator מטפל במסלול
        pass

    def _generate_track_segment(self, count):
        # השבתת יצירת מקטעים
        pass

def run_game(mode="normal", hand_detector=None, show_tutorial=True, config=None):
    """
    Run the game with the specified mode
    
    Args:
        mode: Game mode string ("practice", "easy", "normal", "hard", "time_trial")
        hand_detector: Hand gesture detector instance to use
        show_tutorial: Whether to show the tutorial at the start
        config: Optional mode configuration
        
    Returns:
        Final score
    """
    # Create and initialize the game
    game = Game(hand_detector=hand_detector)
    game.set_mode(mode, config)
    game.show_tutorial = show_tutorial
    
    try:
        # Start the game
        game.start()
        
        # Return the final score
        return game.score_manager.score
    except Exception as e:
        print(f"Error running game: {e}")
        import traceback
        traceback.print_exc()
        return 0
    finally:
        # Clean up
        game.cleanup()


