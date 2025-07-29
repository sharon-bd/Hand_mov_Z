#!/usr/bin/env python3
"""
Standalone SoundManager for testing sound system
"""

import pygame
import time

class SoundManager:
    """Audio management with procedural engine sounds"""
    def __init__(self):
        self.muted = False
        self.engine_channel = None
        self.current_engine_sound = None
        self.last_speed = 0.0
        self.sound_update_time = 0.0
        
        # Initialize pygame mixer for sound
        try:
            # Make sure pygame is initialized first
            if not pygame.get_init():
                pygame.init()
            
            # Initialize mixer with better settings for Windows
            pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=1024)
            pygame.mixer.init()
            
            # Set volume to a reasonable level
            pygame.mixer.set_num_channels(8)  # Allow multiple sounds
            
            print("✅ Sound system initialized successfully")
            print(f"   Mixer settings: {pygame.mixer.get_init()}")
            
            # Test the sound system
            self.test_sound_system()
            
        except Exception as e:
            print(f"⚠️ Sound system initialization failed: {e}")
            import traceback
            traceback.print_exc()
            self.muted = True
    
    def toggle_mute(self):
        """Toggle mute state"""
        self.muted = not self.muted
        if self.muted:
            pygame.mixer.stop()
            print("🔇 Sound muted")
        else:
            print("🔊 Sound unmuted")
        return self.muted
    
    def generate_engine_sound(self, speed, duration=0.2):
        """Generate procedural engine sound based on speed"""
        if self.muted:
            return None
            
        try:
            import numpy as np
            
            # Calculate engine parameters based on speed
            base_freq = 80 + (speed * 200)  # Base frequency: 80-280 Hz
            sample_rate = 22050
            samples = int(sample_rate * duration)
            
            # Create time array
            t = np.linspace(0, duration, samples, False)
            
            # Generate engine sound with multiple harmonics
            wave = np.zeros(samples)
            
            # Add fundamental frequency (engine base note)
            wave += 0.4 * np.sin(2 * np.pi * base_freq * t)
            
            # Add second harmonic (engine character)
            wave += 0.3 * np.sin(2 * np.pi * base_freq * 2 * t)
            
            # Add third harmonic (engine richness)
            wave += 0.2 * np.sin(2 * np.pi * base_freq * 3 * t)
            
            # Add engine roughness (for realism)
            roughness_freq = 20 + (speed * 30)
            wave += 0.1 * np.sin(2 * np.pi * roughness_freq * t)
            
            # Add subtle noise for realism
            noise = np.random.normal(0, 0.05, samples)
            wave += noise
            
            # Apply volume envelope to prevent clicking
            fade_samples = int(0.01 * sample_rate)  # 10ms fade
            if fade_samples > 0:
                wave[:fade_samples] *= np.linspace(0, 1, fade_samples)
                wave[-fade_samples:] *= np.linspace(1, 0, fade_samples)
            
            # Normalize and convert to 16-bit
            wave = np.clip(wave, -1, 1)
            volume = 0.2 + (speed * 0.3)  # Volume increases with speed (reduced volume)
            wave = (wave * volume * 32767).astype(np.int16)
            
            # Convert to stereo - ensure C-contiguous
            stereo_wave = np.column_stack([wave, wave])
            stereo_wave = np.ascontiguousarray(stereo_wave, dtype=np.int16)
            
            # Create pygame sound
            sound = pygame.sndarray.make_sound(stereo_wave)
            sound.set_volume(0.6)  # Set volume on the sound object
            return sound
            
        except ImportError:
            # Fallback if numpy is not available
            print("⚠️ NumPy not available - using simple beep sounds")
            return self._simple_engine_sound(speed, duration)
        except Exception as e:
            print(f"⚠️ Error generating engine sound: {e}")
            return None
    
    def _simple_engine_sound(self, speed, duration=0.2):
        """Simple engine sound fallback without numpy"""
        if self.muted:
            return None
            
        try:
            import math
            
            sample_rate = 22050
            samples = int(sample_rate * duration)
            
            # Generate simple sine wave
            freq = 100 + (speed * 150)
            wave_array = []
            
            for i in range(samples):
                time = float(i) / sample_rate
                value = int(0.2 * 32767 * math.sin(2 * math.pi * freq * time))  # Reduced volume
                wave_array.append([value, value])  # Stereo
            
            # Convert to numpy array for pygame compatibility
            import numpy as np
            wave_array = np.array(wave_array, dtype=np.int16)
            wave_array = np.ascontiguousarray(wave_array)
            
            sound = pygame.sndarray.make_sound(wave_array)
            sound.set_volume(0.5)  # Set volume on the sound object
            return sound
            
        except Exception as e:
            print(f"⚠️ Error generating simple engine sound: {e}")
            return None
    
    def update_engine_sound(self, speed, dt):
        """Update engine sound based on current speed"""
        if self.muted:
            return
            
        current_time = time.time()
        
        # Update sound every 100ms or when speed changes significantly
        speed_change = abs(speed - self.last_speed)
        time_since_update = current_time - self.sound_update_time
        
        if time_since_update > 0.1 or speed_change > 0.1:
            self.last_speed = speed
            self.sound_update_time = current_time
            
            # Stop current engine sound
            if self.engine_channel and self.engine_channel.get_busy():
                self.engine_channel.stop()
            
            # Generate and play new engine sound if car is moving
            if speed > 0.05:  # Only play sound if car is actually moving
                engine_sound = self.generate_engine_sound(speed, 0.15)
                if engine_sound:
                    try:
                        # Stop any existing engine sound first
                        if self.engine_channel and self.engine_channel.get_busy():
                            self.engine_channel.stop()
                            
                        self.engine_channel = pygame.mixer.find_channel()
                        if self.engine_channel:
                            self.engine_channel.play(engine_sound, loops=-1)  # Loop the sound
                            print(f"🔊 Engine sound playing at speed {speed:.2f}")
                        else:
                            channel = engine_sound.play(loops=-1)  # Fixed: use sound.play() instead of pygame.mixer.Sound.play()
                            if channel:
                                print(f"🔊 Engine sound playing (fallback) at speed {speed:.2f}")
                            else:
                                print(f"❌ No channel available for engine sound")
                    except Exception as e:
                        print(f"⚠️ Error playing engine sound: {e}")
                else:
                    print(f"❌ Failed to generate engine sound for speed {speed:.2f}")
            else:
                # Stop engine sound when car stops
                if self.engine_channel and self.engine_channel.get_busy():
                    self.engine_channel.stop()
                    print("🔇 Engine sound stopped (car not moving)")
    
    def play(self, sound_name):
        """Play specific sound effect"""
        if self.muted:
            return
            
        try:
            if sound_name == "collision":
                # Generate collision sound
                crash_sound = self._generate_crash_sound()
                if crash_sound:
                    crash_sound.play()  # Fixed: use sound.play() instead of pygame.mixer.Sound.play()
                    print("🔊 Collision sound played")
            elif sound_name == "boost":
                # Generate boost sound
                boost_sound = self._generate_boost_sound()
                if boost_sound:
                    boost_sound.play()  # Fixed: use sound.play() instead of pygame.mixer.Sound.play()
                    print("🔊 Boost sound played")
        except Exception as e:
            print(f"⚠️ Error playing sound {sound_name}: {e}")
    
    def _generate_crash_sound(self):
        """Generate crash sound effect"""
        try:
            import numpy as np
            
            duration = 0.3
            sample_rate = 22050
            samples = int(sample_rate * duration)
            
            # Generate crash noise
            noise = np.random.normal(0, 0.8, samples)
            
            # Apply envelope for crash effect
            envelope = np.exp(-np.linspace(0, 5, samples))
            wave = noise * envelope
            
            # Convert to 16-bit stereo - ensure C-contiguous
            wave = np.clip(wave, -1, 1)
            wave = (wave * 0.6 * 32767).astype(np.int16)
            stereo_wave = np.column_stack([wave, wave])
            stereo_wave = np.ascontiguousarray(stereo_wave, dtype=np.int16)
            
            sound = pygame.sndarray.make_sound(stereo_wave)
            sound.set_volume(0.7)
            return sound
        except:
            return None
    
    def _generate_boost_sound(self):
        """Generate boost sound effect"""
        try:
            import numpy as np
            
            duration = 0.5
            sample_rate = 22050
            samples = int(sample_rate * duration)
            t = np.linspace(0, duration, samples, False)
            
            # Rising frequency for boost effect
            freq = 200 + 400 * t / duration
            wave = 0.4 * np.sin(2 * np.pi * freq * t)
            
            # Apply envelope
            envelope = np.exp(-t * 2)
            wave *= envelope
            
            # Convert to 16-bit stereo - ensure C-contiguous
            wave = np.clip(wave, -1, 1)
            wave = (wave * 32767).astype(np.int16)
            stereo_wave = np.column_stack([wave, wave])
            stereo_wave = np.ascontiguousarray(stereo_wave, dtype=np.int16)
            
            sound = pygame.sndarray.make_sound(stereo_wave)
            sound.set_volume(0.6)
            return sound
        except:
            return None
    
    def test_sound_system(self):
        """Test if sound system is working"""
        if self.muted:
            print("🔇 Sound system is muted")
            return False
            
        try:
            print("🔊 Testing sound system...")
            
            # Test simple beep
            import math
            duration = 0.5
            sample_rate = 22050
            frequency = 440
            samples = int(sample_rate * duration)
            
            wave_array = []
            for i in range(samples):
                time_point = float(i) / sample_rate
                value = int(0.3 * 32767 * math.sin(2 * math.pi * frequency * time_point))
                wave_array.append([value, value])
            
            # Convert to numpy array for pygame compatibility
            import numpy as np
            wave_array = np.array(wave_array, dtype=np.int16)
            wave_array = np.ascontiguousarray(wave_array)
            
            test_sound = pygame.sndarray.make_sound(wave_array)
            test_sound.set_volume(0.5)  # Set volume to 50%
            channel = test_sound.play()
            
            if channel:
                print("✅ Sound system working - test beep played")
                return True
            else:
                print("❌ Sound system not working - no channel available")
                return False
                
        except Exception as e:
            print(f"❌ Sound system test failed: {e}")
            return False
    
    def cleanup(self):
        """Clean up sound resources"""
        try:
            if self.engine_channel:
                self.engine_channel.stop()
            pygame.mixer.quit()
            print("✅ Sound system cleaned up")
        except Exception as e:
            print(f"⚠️ Error cleaning up sound: {e}")


def main():
    """Test the standalone sound manager"""
    print("🎵 STANDALONE SOUND MANAGER TEST")
    print("=" * 50)
    
    # Create sound manager
    sound_manager = SoundManager()
    
    if sound_manager.muted:
        print("❌ Sound manager failed to initialize")
        return
    
    print("\n🔊 Testing different engine speeds...")
    
    speeds = [0.0, 0.2, 0.5, 0.8, 1.0]
    for speed in speeds:
        print(f"\nTesting speed {speed:.1f}...")
        engine_sound = sound_manager.generate_engine_sound(speed, 1.0)
        if engine_sound:
            print(f"🔊 Playing engine sound at speed {speed:.1f}")
            channel = engine_sound.play()
            time.sleep(1.2)  # Let sound play
        else:
            print(f"❌ Failed to generate sound for speed {speed:.1f}")
    
    print("\n🔊 Testing sound effects...")
    
    # Test collision sound
    print("Testing collision sound...")
    sound_manager.play("collision")
    time.sleep(1.5)
    
    # Test boost sound  
    print("Testing boost sound...")
    sound_manager.play("boost")
    time.sleep(1.5)
    
    print("\n🔊 Testing engine update method...")
    
    # Test the update method like in the game
    for i in range(10):
        speed = 0.1 + (i * 0.1)
        sound_manager.update_engine_sound(speed, 0.1)
        time.sleep(0.2)
    
    # Stop engine
    sound_manager.update_engine_sound(0.0, 0.1)
    time.sleep(1)
    
    # Cleanup
    sound_manager.cleanup()
    
    print("\n✅ Sound manager test completed!")
    print("If you heard sounds, the system is working correctly.")
    print("If not, check your system volume and audio drivers.")
    
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
