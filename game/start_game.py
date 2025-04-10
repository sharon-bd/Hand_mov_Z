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
        """
        Initialize the game
        
        Args:
            screen_width: Width of the game screen
            screen_height: Height of the game screen
            hand_detector: Hand gesture detector instance to use
        """
        # Initialize pygame
        if not pygame.get_init():
            pygame.init()
        
        # Screen setup
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Hand Gesture Car Control")
        
        # Clock for controlling frame rate
        self.clock = pygame.time.Clock()
        self.fps = 60
        
        # Game state
        self.running = False
        self.paused = False
        self.game_over = False
        self.current_mode = "normal"  # Default mode
        
        # Time tracking
        self.start_time = 0
        self.elapsed_time = 0
        self.time_limit = None
        
        # Initialize game components
        self.renderer = GameRenderer(screen_width, screen_height)
        self.physics = PhysicsEngine()
        self.audio = AudioManager()
        self.camera = CameraManager()
        self.score_manager = ScoreManager()
        
        # Road parameters (for obstacle positioning)
        self.road_width = self.renderer.road_width
        self.road_x = self.renderer.road_x
        
        # Create managers
        self.obstacle_manager = ObstacleManager(
            screen_width, screen_height, self.road_width, self.road_x
        )
        self.power_up_manager = PowerUpManager(
            screen_width, screen_height, self.road_width, self.road_x
        )
        
        # Create the player's car
        self.car = Car(screen_width // 2, screen_height - 100)
        
        # Hand detector
        self.hand_detector = hand_detector
        
        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Tutorial state
        self.show_tutorial = False
        self.tutorial_step = 0
        self.tutorial_steps = [
            "Welcome to Hand Gesture Car Control!",
            "Move your hand left and right to steer the car.",
            "Move your hand up and down to control speed.",
            "Make a fist to brake. Keep your fist raised for boost!",
            "Show an open palm (stop sign) to emergency stop.",
            "Avoid obstacles and collect power-ups to score points.",
            "Ready to begin? Press SPACE to start."
        ]
    
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
        """Main game loop"""
        while self.running:
            # Calculate time delta
            dt = self.clock.tick(self.fps) / 1000.0
            
            # Process events
            self.handle_events()
            
            # Skip updates if paused
            if not self.paused:
                # Update game state
                self.update(dt)
                
                # Check for game over conditions
                self._check_game_over()
            
            # Draw everything
            self.draw()
            
            # Update the display
            pygame.display.flip()
    
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