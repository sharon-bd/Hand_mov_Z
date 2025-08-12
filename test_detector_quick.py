#!/usr/bin/env python3
"""
Quick test for hand gesture detector without camera display
"""

import cv2
import numpy as np
from hand_detector.simple_detector import SimpleHandGestureDetector

def test_detector():
    """Test the detector with a simple frame"""
    print("Testing SimpleHandGestureDetector...")
    
    # Create detector
    detector = SimpleHandGestureDetector()
    print("✅ Detector created successfully")
    
    # Create a test frame (black image)
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    print("✅ Test frame created")
    
    # Test the detect_gestures method
    try:
        controls, processed_frame, data_panel = detector.detect_gestures(test_frame)
        print("✅ detect_gestures method works correctly")
        print(f"Controls returned: {controls}")
        print(f"Processed frame shape: {processed_frame.shape}")
        print(f"Data panel shape: {data_panel.shape}")
        return True
    except Exception as e:
        print(f"❌ Error in detect_gestures: {e}")
        return False

if __name__ == "__main__":
    test_detector()
