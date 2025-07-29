#!/usr/bin/env python
"""
Debug Utilities for Hand Gesture Car Control System

This module provides debugging tools to help with development and troubleshooting.
"""

import os
import sys
import time
import pygame
import cv2
import numpy as np

class PerformanceMonitor:
    """
    Monitors and displays performance metrics for the application
    """
    
    def __init__(self, window_size=60):
        """
        Initialize the performance monitor
        
        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.frame_times = []
        self.detection_times = []
        self.render_times = []
        self.update_times = []
        self.start_time = time.time()
        self.current_frame = 0
    
    def start_frame(self):
        """Start timing a new frame"""
        self.start_time = time.time()
        self.current_frame += 1
    
    def record_detection(self):
        """Record time taken for hand detection"""
        current_time = time.time()
        self.detection_times.append(current_time - self.start_time)
        self.start_time = current_time
        
        # Keep only the last window_size entries
        if len(self.detection_times) > self.window_size:
            self.detection_times.pop(0)
    
    def record_update(self):
        """Record time taken for game state update"""
        current_time = time.time()
        self.update_times.append(current_time - self.start_time)
        self.start_time = current_time
        
        # Keep only the last window_size entries
        if len(self.update_times) > self.window_size:
            self.update_times.pop(0)
    
    def record_render(self):
        """Record time taken for rendering"""
        current_time = time.time()
        self.render_times.append(current_time - self.start_time)
        self.start_time = current_time
        
        # Keep only the last window_size entries
        if len(self.render_times) > self.window_size:
            self.render_times.pop(0)
    
    def end_frame(self):
        """End timing the current frame"""
        current_time = time.time()
        self.frame_times.append(current_time - self.start_time)
        
        # Keep only the last window_size entries
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
    
    def get_stats(self):
        """
        Get performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        stats = {
            'frame_rate': 0,
            'detection_time': 0,
            'update_time': 0,
            'render_time': 0,
            'total_frame_time': 0
        }
        
        if self.frame_times:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            stats['frame_rate'] = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            stats['total_frame_time'] = avg_frame_time * 1000  # Convert to ms
        
        if self.detection_times:
            stats['detection_time'] = sum(self.detection_times) / len(self.detection_times) * 1000  # ms
        
        if self.update_times:
            stats['update_time'] = sum(self.update_times) / len(self.update_times) * 1000  # ms
        
        if self.render_times:
            stats['render_time'] = sum(self.render_times) / len(self.render_times) * 1000  # ms
        
        return stats
    
    def draw_overlay(self, surface, position=(10, 10), font_size=20):
        """
        Draw performance stats overlay on the given surface
        
        Args:
            surface: Pygame surface to draw on
            position: Position to draw the overlay
            font_size: Font size for the text
        """
        stats = self.get_stats()
        
        try:
            font = pygame.font.Font(None, font_size)
        except:
            # If pygame font is not initialized
            return
        
        lines = [
            f"FPS: {stats['frame_rate']:.1f}",
            f"Frame Time: {stats['total_frame_time']:.1f}ms",
            f"Detection: {stats['detection_time']:.1f}ms",
            f"Update: {stats['update_time']:.1f}ms",
            f"Render: {stats['render_time']:.1f}ms"
        ]
        
        # Draw background
        line_height = font_size + 2
        box_height = line_height * len(lines) + 10
        box_width = 200
        
        pygame.draw.rect(
            surface,
            (0, 0, 0, 180),
            (position[0], position[1], box_width, box_height),
            0,
            5
        )
        
        # Draw text
        y = position[1] + 5
        for line in lines:
            text = font.render(line, True, (255, 255, 255))
            surface.blit(text, (position[0] + 5, y))
            y += line_height


class DebugVisualizer:
    """
    Provides visualization tools for debugging
    """
    
    @staticmethod
    def create_heatmap(points, width, height, radius=20):
        """
        Create a heatmap image from a set of points
        
        Args:
            points: List of (x, y) tuples
            width, height: Dimensions of the output image
            radius: Radius of influence for each point
            
        Returns:
            Numpy array representing the heatmap image
        """
        # Create an empty heatmap
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        # Add each point to the heatmap
        for x, y in points:
            if 0 <= x < width and 0 <= y < height:
                # Create a circular area of influence
                y_min = max(0, int(y - radius))
                y_max = min(height, int(y + radius))
                x_min = max(0, int(x - radius))
                x_max = min(width, int(x + radius))
                
                for py in range(y_min, y_max):
                    for px in range(x_min, x_max):
                        # Calculate distance from point
                        dist = np.sqrt((px - x)**2 + (py - y)**2)
                        if dist <= radius:
                            # Add value based on distance (closer = higher value)
                            value = 1.0 - (dist / radius)
                            heatmap[py, px] += value
        
        # Normalize heatmap
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        # Convert to color image (jet colormap)
        heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        return heatmap_color
    
    @staticmethod
    def overlay_heatmap(image, heatmap, alpha=0.5):
        """
        Overlay a heatmap on an image
        
        Args:
            image: Base image
            heatmap: Heatmap image
            alpha: Transparency of the heatmap
            
        Returns:
            Image with heatmap overlay
        """
        # Resize heatmap if needed
        if image.shape[:2] != heatmap.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Blend images
        return cv2.addWeighted(image, 1.0, heatmap, alpha, 0)
    
    @staticmethod
    def draw_trajectory(surface, points, color=(0, 255, 0), thickness=2):
        """
        Draw a trajectory line connecting points
        
        Args:
            surface: Pygame surface to draw on
            points: List of (x, y) tuples
            color: Line color
            thickness: Line thickness
        """
        if len(points) < 2:
            return
        
        # Draw lines connecting points
        for i in range(len(points) - 1):
            pygame.draw.line(
                surface,
                color,
                points[i],
                points[i + 1],
                thickness
            )


class LoggingManager:
    """
    Manages logging for the application
    """
    
    def __init__(self, log_file="game_log.log", log_level="INFO"):
        """
        Initialize the logging manager
        
        Args:
            log_file: Path to the log file
            log_level: Minimum log level to record
        """
        self.log_file = log_file
        self.log_level = log_level
        self.log_levels = {
            "DEBUG": 0,
            "INFO": 1,
            "WARNING": 2,
            "ERROR": 3,
            "CRITICAL": 4
        }
        
        # Create log file
        with open(self.log_file, 'w') as f:
            f.write(f"=== Hand Gesture Car Control System Log ===\n")
            f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    def log(self, message, level="INFO"):
        """
        Log a message
        
        Args:
            message: Message to log
            level: Log level
        """
        if self.log_levels.get(level, 0) >= self.log_levels.get(self.log_level, 0):
            timestamp = time.strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {level}: {message}"
            
            # Print to console
            print(log_entry)
            
            # Write to log file
            with open(self.log_file, 'a') as f:
                f.write(log_entry + "\n")
    
    def debug(self, message):
        """Log a debug message"""
        self.log(message, "DEBUG")
    
    def info(self, message):
        """Log an info message"""
        self.log(message, "INFO")
    
    def warning(self, message):
        """Log a warning message"""
        self.log(message, "WARNING")
    
    def error(self, message):
        """Log an error message"""
        self.log(message, "ERROR")
    
    def critical(self, message):
        """Log a critical message"""
        self.log(message, "CRITICAL")