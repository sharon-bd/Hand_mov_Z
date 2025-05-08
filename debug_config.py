"""
Debug configuration for the Hand Movement Project

This file controls debug settings across the application.
"""

# Global settings
DEBUG_ENABLED = True  # Set to True to enable debug output
DEBUG_LEVEL = 1        # 0=None, 1=Basic, 2=Detailed, 3=Verbose

# Component-specific debug settings
COMPONENTS = {
    'car': True,               # Car physics debugging - enable to see car physics info
    'camera': False,           # Camera feed debugging  
    'detector': True,          # Hand gesture detector debugging
    'controls': True,          # Control values debugging - enable to track steering issues
    'performance': True,       # FPS and performance metrics
}

# Debugging frequency limits (in seconds) to prevent spam
FREQUENCY_LIMITS = {
    'controls': 0.5,           # Log more frequently to track steering issues (was 1.0)
    'camera': 5.0,             # Camera logs limited to once per 5 seconds
    'performance': 10.0,       # Performance logs every 10 seconds
    'car': 1.0,                # Car physics logging frequency
}

# Track last log times for frequency limiting
_last_log_times = {}
# Track last logged values to only log when they change
_last_values = {}

def is_debug_enabled(component=None):
    """
    Check if debugging is enabled for a specific component
    
    Args:
        component: Component name to check, or None for global setting
        
    Returns:
        Boolean indicating if debug is enabled
    """
    if not DEBUG_ENABLED:
        return False
        
    if component is None:
        return DEBUG_ENABLED
        
    return COMPONENTS.get(component, False)

def can_log(component, value=None):
    """
    Check if we should log for this component based on frequency limits and value changes
    
    Args:
        component: Component name to check
        value: The value to log - if provided, will only log if changed
        
    Returns:
        Boolean indicating if logging is allowed at this time
    """
    import time
    
    if not is_debug_enabled(component):
        return False
        
    current_time = time.time()
    last_time = _last_log_times.get(component, 0)
    limit = FREQUENCY_LIMITS.get(component, 0)
    
    # Check if value changed (if provided)
    value_changed = True
    if value is not None:
        last_value = _last_values.get(component)
        value_changed = last_value != value
        if value_changed:
            _last_values[component] = value
    
    # Only log if minimum time passed or value changed
    if (current_time - last_time >= limit) or value_changed:
        _last_log_times[component] = current_time
        return True
        
    return False

def log(component, message, level=1, value=None):
    """
    Log a debug message if the component is enabled and level is sufficient
    
    Args:
        component: Component name for the log
        message: Message to log
        level: Debug level (1-3)
        value: Value being logged, to avoid duplicate logs
    """
    if level > DEBUG_LEVEL:
        return
        
    if not can_log(component, value):
        return
        
    import time
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] DEBUG - {component}: {message}")
