#!/usr/bin/env python3
"""
Minimal test for basic imports only
"""

def test_basic_imports():
    """Test basic imports step by step"""
    print("Starting basic import test...")
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except Exception as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except Exception as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import mediapipe as mp
        print("✅ MediaPipe imported successfully")
        print(f"MediaPipe version: {mp.__version__}")
    except Exception as e:
        print(f"❌ MediaPipe import failed: {e}")
        return False
    
    print("✅ All basic imports successful!")
    return True

if __name__ == "__main__":
    test_basic_imports()
