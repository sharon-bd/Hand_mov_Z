#!/usr/bin/env python
"""
Audio Manager for Hand Gesture Car Control Game

This module handles all sound effects and music for the game.
"""

import os
import pygame
import math
import random
import time

class AudioManager:
    """Manages all sound effects and music for the game"""
    
    def __init__(self):
        """Initialize the audio manager"""
        # Initialize pygame mixer
        pygame.mixer.init()
        
        # Sound settings
        self.sound_enabled = True
        self.music_enabled = True
        self.volume = 0.7
        self.music_volume = 0.5
        
        # Dictionary to store sound effects
        self.sounds = {}
        
        # Engine sound state
        self.engine_channel = None
        self.current_engine_pitch = 1.0
        self.target_engine_pitch = 1.0
        
        # Load sound effects
        self._load_sounds()
    
    def _load_sounds(self):
        """Load all sound effects"""
        # Define sound file paths
        sound_files = {
            'engine_idle': 'sounds/engine_idle.wav',
            'engine_revving': 'sounds/engine_revving.wav',
            'collision': 'sounds/collision.wav',
            'boost': 'sounds/boost.wav',
            'brake': 'sounds/brake.wav',
            'power_up': 'sounds/power_up.wav',
            'menu_select': 'sounds/menu_select.wav',
            'menu_move': 'sounds/menu_move.wav'
        }
        
        # Create sounds directory if it doesn't exist
        if not os.path.exists('sounds'):
            os.makedirs('sounds')
        
        # Load each sound or generate a substitute if file missing
        for name, path in sound_files.items():
            try:
                if os.path.exists(path):
                    self.sounds[name] = pygame.mixer.Sound(path)
                else:
                    print(f"Warning: Sound file not found: {path}")
                    self.sounds[name] = self._generate_substitute_sound(name)
            except Exception as e:
                print(f"Error loading sound {name}: {e}")
                # Generate a substitute sound
                self.sounds[name] = self._generate_substitute_sound(name)
        
        # Set volumes
        for sound in self.sounds.values():
            sound.set_volume(self.volume)
    
    def _generate_substitute_sound(self, name):
        """Generate a substitute sound effect if the file is missing"""
        # Create a short sound buffer
        sample_rate = 44100
        duration = 1.0  # seconds
        
        if 'engine' in name:
            # Generate engine sound (low rumble)
            buf = self._generate_engine_sound(sample_rate, duration)
        elif 'collision' in name:
            # Generate collision sound (noise burst)
            buf = self._generate_noise_burst(sample_rate, 0.3)
        elif 'boost' in name:
            # Generate boost sound (rising tone)
            buf = self._generate_rising_tone(sample_rate, 0.5)
        elif 'brake' in name:
            # Generate brake sound (short noise)
            buf = self._generate_noise_burst(sample_rate, 0.2)
        elif 'power_up' in name:
            # Generate power-up sound (ascending ding)
            buf = self._generate_ding_sound(sample_rate, 0.3)
        elif 'menu' in name:
            # Generate menu sound (short beep)
            buf = self._generate_beep_sound(sample_rate, 0.1)
        else:
            # Default to a simple beep
            buf = self._generate_beep_sound(sample_rate, 0.2)
        
        # Create a Sound object from the buffer
        sound = pygame.mixer.Sound(buffer=buf)
        return sound
    
    def _generate_engine_sound(self, sample_rate, duration):
        """Generate a basic engine rumble sound"""
        num_samples = int(sample_rate * duration)
        buf = bytearray(num_samples)
        
        # Create a low rumble by mixing sine waves
        for i in range(num_samples):
            t = i / sample_rate
            
            # Mix multiple frequencies
            value = 0
            value += 0.3 * math.sin(2 * math.pi * 50 * t)  # 50 Hz base
            value += 0.2 * math.sin(2 * math.pi * 100 * t)  # 100 Hz harmonic
            value += 0.1 * math.sin(2 * math.pi * 150 * t)  # 150 Hz harmonic
            value += 0.1 * math.sin(2 * math.pi * 75 * t)  # 75 Hz harmonic
            
            # Add some noise
            value += 0.1 * random.uniform(-1, 1)
            
            # Convert to byte range (0-255)
            buf[i] = max(0, min(255, int(128 + value * 60)))
        
        return buf
    
    def _generate_noise_burst(self, sample_rate, duration):
        """Generate a noise burst for collision sounds"""
        num_samples = int(sample_rate * duration)
        buf = bytearray(num_samples)
        
        # Create a burst of noise that fades out
        for i in range(num_samples):
            t = i / sample_rate
            fade = 1.0 - (t / duration)
            
            # Random noise with fade-out
            value = random.uniform(-1, 1) * fade
            
            # Convert to byte range (0-255)
            buf[i] = max(0, min(255, int(128 + value * 127)))
        
        return buf
    
    def _generate_rising_tone(self, sample_rate, duration):
        """Generate a rising tone for boost sounds"""
        num_samples = int(sample_rate * duration)
        buf = bytearray(num_samples)
        
        # Create a rising tone
        for i in range(num_samples):
            t = i / sample_rate
            
            # Frequency rises from 300Hz to 1200Hz
            freq = 300 + 900 * (t / duration)
            
            # Amplitude with a slight fade-in and fade-out
            fade = min(t * 10, (duration - t) * 10, 1.0)
            
            value = math.sin(2 * math.pi * freq * t) * fade
            
            # Convert to byte range (0-255)
            buf[i] = max(0, min(255, int(128 + value * 127)))
        
        return buf
    
    def _generate_ding_sound(self, sample_rate, duration):
        """Generate a ding sound for power-ups"""
        num_samples = int(sample_rate * duration)
        buf = bytearray(num_samples)
        
        # Create a ding sound (two rising tones)
        for i in range(num_samples):
            t = i / sample_rate
            
            # First tone
            freq1 = 800
            value1 = math.sin(2 * math.pi * freq1 * t)
            
            # Second tone with rising frequency
            freq2 = 1200 + 400 * (t / duration)
            value2 = math.sin(2 * math.pi * freq2 * t)
            
            # Mix the tones with a fade-out
            fade = 1.0 - (t / duration)**2
            value = (0.5 * value1 + 0.5 * value2) * fade
            
            # Convert to byte range (0-255)
            buf[i] = max(0, min(255, int(128 + value * 127)))
        
        return buf
    
    def _generate_beep_sound(self, sample_rate, duration):
        """Generate a simple beep sound for menu effects"""
        num_samples = int(sample_rate * duration)
        buf = bytearray(num_samples)
        
        # Create a simple beep
        for i in range(num_samples):
            t = i / sample_rate
            
            # Simple sine wave
            freq = 800
            fade = min(t * 20, (duration - t) * 20, 1.0)  # Quick fade in/out
            
            value = math.sin(2 * math.pi * freq * t) * fade
            
            # Convert to byte range (0-255)
            buf[i] = max(0, min(255, int(128 + value * 127)))
        
        return buf
    
    def set_sound_enabled(self, enabled):
        """Enable or disable all sound effects"""
        self.sound_enabled = enabled
        
        # Stop all sounds if disabled
        if not enabled:
            pygame.mixer.stop()
        
        # Resume engine sound if enabled
        elif self.engine_channel is not None:
            self._play_engine_sound(self.current_engine_pitch)
    
    def set_music_enabled(self, enabled):
        """Enable or disable background music"""
        self.music_enabled = enabled
        
        if enabled:
            self._resume_music()
        else:
            pygame.mixer.music.pause()
    
    def set_volume(self, volume):
        """Set the volume for sound effects"""
        self.volume = max(0.0, min(1.0, volume))
        
        # Update volume for all sounds
        for sound in self.sounds.values():
            sound.set_volume(self.volume)
    
    def set_music_volume(self, volume):
        """Set the volume for background music"""
        self.music_volume = max(0.0, min(1.0, volume))
        pygame.mixer.music.set_volume(self.music_volume)
    
    def play_sound(self, name):
        """Play a sound by name"""
        if not self.sound_enabled:
            return
        
        if name in self.sounds:
            self.sounds[name].play()
    
    def update_engine_sound(self, speed, boost=False):
        """
        Update the engine sound based on car speed
        
        Args:
            speed: Current car speed (0.0 to 1.0)
            boost: Whether boost is active
        """
        if not self.sound_enabled:
            return
        
        # Calculate target pitch based on speed
        base_pitch = 0.5 + speed * 0.75
        
        # Add boost effect
        if boost:
            base_pitch *= 1.3
        
        self.target_engine_pitch = base_pitch
        
        # Initialize engine sound channel if needed
        if self.engine_channel is None:
            self._play_engine_sound(self.target_engine_pitch)
        else:
            # Smoothly adjust pitch
            self.current_engine_pitch = self.current_engine_pitch + (
                self.target_engine_pitch - self.current_engine_pitch) * 0.1
            
            # Only update if change is significant
            if abs(self.current_engine_pitch - self.engine_channel.get_busy()) > 0.05:
                self._play_engine_sound(self.current_engine_pitch)
    
    def _play_engine_sound(self, pitch):
        """Play the engine sound at the specified pitch"""
        if 'engine_revving' not in self.sounds:
            return
        
        # Stop current engine sound
        if self.engine_channel is not None:
            self.engine_channel.stop()
        
        # Play the engine sound with the specified pitch
        sound = self.sounds['engine_revving']
        self.engine_channel = sound.play(-1)  # Loop indefinitely
        
        if self.engine_channel is not None:
            # Some Pygame versions don't support this, so we check
            if hasattr(self.engine_channel, 'set_volume'):
                self.engine_channel.set_volume(self.volume * 0.7)  # Slightly lower volume for engine
                
            # Some Pygame versions support changing pitch
            if hasattr(pygame.mixer, 'Sound') and hasattr(pygame.mixer.Sound, 'get_raw'):
                try:
                    self.engine_channel.set_pitch(pitch)
                except AttributeError:
                    pass  # Pitch control not available
    
    def play_collision_sound(self):
        """Play the collision sound effect"""
        self.play_sound('collision')
    
    def play_boost_sound(self):
        """Play the boost sound effect"""
        self.play_sound('boost')
    
    def play_brake_sound(self):
        """Play the brake sound effect"""
        self.play_sound('brake')
    
    def play_power_up_sound(self):
        """Play the power-up sound effect"""
        self.play_sound('power_up')
    
    def play_menu_select_sound(self):
        """Play the menu selection sound effect"""
        self.play_sound('menu_select')
    
    def play_menu_move_sound(self):
        """Play the menu movement sound effect"""
        self.play_sound('menu_move')
    
    def load_background_music(self, filename):
        """
        Load and play background music
        
        Args:
            filename: Path to the music file
        """
        if not self.music_enabled:
            return
        
        try:
            if os.path.exists(filename):
                pygame.mixer.music.load(filename)
                pygame.mixer.music.set_volume(self.music_volume)
                pygame.mixer.music.play(-1)  # Loop indefinitely
            else:
                print(f"Warning: Music file not found: {filename}")
        except Exception as e:
            print(f"Error loading music: {e}")
    
    def _resume_music(self):
        """Resume previously loaded background music"""
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.unpause()
    
    def stop_music(self):
        """Stop the background music"""
        pygame.mixer.music.stop()
    
    def cleanup(self):
        """Clean up resources"""
        # Stop all sounds
        pygame.mixer.stop()
        
        # Stop music
        pygame.mixer.music.stop()