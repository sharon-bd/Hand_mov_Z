import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import cv2
import time
import numpy as np
from hand_detector.improved_hand_gesture_detector_fixed import EnhancedHandGestureDetector
from hand_detector.gestures import HandGestureDetector 
from hand_detector.simple_detector import SimpleHandGestureDetector  # Add import

def test_hand_detector(detector_class='improved'):
    """
    Test hand gesture detector with webcam input
    
    Args:
        detector_class: 'improved' for EnhancedHandGestureDetector, 'basic' for HandGestureDetector, or 'simple' for SimpleHandGestureDetector
    """
    print(f"Testing {detector_class} hand gesture detector...")
    
    # Initialize the detector
    if detector_class == 'improved':
        detector = EnhancedHandGestureDetector()
    elif detector_class == 'basic':
        detector = HandGestureDetector()
    else:
        detector = SimpleHandGestureDetector()
    
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Verify webcam initialization
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Check actual resolution
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Webcam resolution: {actual_width}x{actual_height}")
    if actual_width != 640 or actual_height != 480:
        print("Warning: Webcam resolution does not match requested 640x480")
    
    # FPS calculation variables
    prev_frame_time = 0
    new_frame_time = 0
    
    try:
        while True:
            # Read frame from webcam
            success, frame = cap.read()
            if not success:
                print("Error: Could not read frame from webcam")
                break
            
            # Calculate FPS
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 30
            prev_frame_time = new_frame_time
            fps_text = f"FPS: {fps:.1f}"
            
            # Process frame with detector
            if detector_class == 'improved':
                controls, processed_frame, data_panel = detector.detect_gestures(frame)
                # Add margin at bottom, but do NOT draw FPS here (already drawn by detector)
                margin = 80
                h, w = processed_frame.shape[:2]
                frame_with_margin = np.zeros((h + margin, w, 3), dtype=np.uint8)
                frame_with_margin[:h, :] = processed_frame
                # Do NOT draw FPS here
                cv2.imshow('Hand Gestures', frame_with_margin)
                cv2.imshow('Data Analysis', data_panel)
            elif detector_class == 'basic':
                # Basic detector
                controls, processed_frame = detector.detect_gestures(frame)
                cv2.putText(processed_frame, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 255, 0), 2)
                cv2.imshow('Hand Gestures', processed_frame)
            else:
                # Simple detector
                controls, processed_frame = detector.detect_gestures(frame)
                cv2.putText(processed_frame, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 255, 0), 2)
                cv2.imshow('Hand Gestures', processed_frame)
            
            # Print control values
            print(f"\rSteering: {controls['steering']:.2f}, Throttle: {controls['throttle']:.2f}, "
                  f"Gesture: {controls['gesture_name']}", end='', flush=True)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("\nTest finished.")

def create_hand_visualization():
    """Create a separate visualization window for hand gesture data"""
    detector = SimpleHandGestureDetector()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    prev_frame_time = 0
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # Calculate FPS and process frame
            current_time = time.time()
            fps = 1 / (current_time - prev_frame_time) if prev_frame_time > 0 else 0
            prev_frame_time = current_time
            
            detector.fps = fps
            controls, processed_frame, data_panel = detector.detect_gestures(cv2.flip(frame, 1))
            
            # Create combined visualization
            h, w = processed_frame.shape[:2]
            margin = 40
            combined_frame = np.zeros((h + 80 + margin, w, 3), dtype=np.uint8)
            combined_frame[:h, :] = processed_frame
            cv2.line(combined_frame, (0, h), (w, h), (200, 200, 200), 2)
            
            # Show FPS in top-left
            cv2.putText(combined_frame, f"FPS: {fps:.1f}", 
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('Hand Gesture Visualization', combined_frame)
            cv2.imshow('Control Data', data_panel)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Hand Gesture Detection Test with Fixed UI")
    print("Options:")
    print("1. Test with improved detector")
    print("2. Test with basic detector")
    print("3. Test with simple detector")
    print("4. Run enhanced visualization")
    
    try:
        choice = int(input("Select option (1-4): "))
        if choice == 1:
            test_hand_detector('improved')
        elif choice == 2:
            test_hand_detector('basic')
        elif choice == 3:
            test_hand_detector('simple')
        elif choice == 4:
            create_hand_visualization()
        else:
            print("Invalid choice. Using simple detector.")
            test_hand_detector('simple')
    except ValueError:
        print("Invalid input. Using simple detector.")
        test_hand_detector('simple')