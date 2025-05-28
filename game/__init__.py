# Game package initialization file
"""
Hand Gesture Car Control Game Package

This package contains all the game modules including:
- start_game: Main game logic
- moving_road: Advanced road animation system
"""

__version__ = "1.0.0"
__author__ = "Hand Gesture Car Control Team"

# Import main components for easy access
try:
    from .start_game import run_game, Game
    from .moving_road import MovingRoadGenerator
    
    __all__ = ['run_game', 'Game', 'MovingRoadGenerator']
    
    print("✅ Game package initialized successfully")
    
except ImportError as e:
    print(f"⚠️ Warning: Some game modules could not be imported: {e}")
    
    # Fallback imports
    try:
        from .start_game import run_game, Game
        __all__ = ['run_game', 'Game']
        print("✅ Core game modules loaded")
    except ImportError:
        print("❌ Critical: Could not load core game modules")
        __all__ = []