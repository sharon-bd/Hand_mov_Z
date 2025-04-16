import cv2
import time
from hand_detector.improved_hand_gesture_detector_fixed import EnhancedHandGestureDetector
from hand_detector.gestures import HandGestureDetector 

def test_hand_detector(detector_class='improved'):
    """
    Test hand gesture detector with webcam input
    
    Args:
        detector_class: 'improved' for EnhancedHandGestureDetector or 'basic' for HandGestureDetector
    """
    print(f"Testing {detector_class} hand gesture detector...")
    
    # Initialize the detector
    if detector_class == 'improved':
        detector = EnhancedHandGestureDetector()
    else:
        detector = HandGestureDetector()
    
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
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
                # Display the processed frame and data panel side by side
                cv2.putText(processed_frame, fps_text, (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0, 255, 0), 2)
                
                # Show frame and data panel
                cv2.imshow('Hand Gestures', processed_frame)
                cv2.imshow('Data Analysis', data_panel)
            else:
                # Basic detector
                controls, processed_frame = detector.detect_gestures(frame)
                cv2.putText(processed_frame, fps_text, (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 
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

if __name__ == "__main__":
    # Usage: Change 'improved' to 'basic' to test the basic detector instead
    test_hand_detector('improved')