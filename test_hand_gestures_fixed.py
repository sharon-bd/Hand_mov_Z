import cv2
import numpy as np
import time
from hand_detector.improved_hand_gesture_detector_fixed import EnhancedHandGestureDetector
import os

def list_available_cameras():
    """Try to detect available cameras on the system"""
    available_cameras = []
    for i in range(10):  # Check first 10 camera indices
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DirectShow API for Windows
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
                print(f"Camera {i} is available - Resolution: {frame.shape[1]}x{frame.shape[0]}")
            cap.release()
        else:
            print(f"Camera {i} is not available")
    return available_cameras

def test_camera_access():
    """Test if we can access the webcam and print diagnostics"""
    print("Testing camera access...")
    
    # First try with DirectShow on Windows
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("Failed to open camera with DirectShow, trying default API...")
        cap = cv2.VideoCapture(0)  # Try with default API
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Successfully accessed camera - Resolution: {frame.shape[1]}x{frame.shape[0]}")
            cap.release()
            return True
        else:
            print("Camera opened but failed to read frame")
            cap.release()
    else:
        print("Failed to access any camera")
    
    return False

def get_video_source():
    """Get video source based on user preference and availability"""
    
    # Check if webcam access works
    camera_works = test_camera_access()
    
    if camera_works:
        print("\nWebcam is accessible.")
        choice = input("Use webcam? (y/n): ").lower()
        if choice == 'y':
            # Try DirectShow first for Windows
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)  # Fall back to default
            return cap, "webcam"
    else:
        print("\nWebcam is not accessible.")
    
    # If webcam isn't working or user doesn't want to use it, offer sample videos
    sample_dir = os.path.join(os.path.dirname(__file__), "sample_videos")
    
    # Create directory if it doesn't exist
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
        print(f"Created directory for sample videos: {sample_dir}")
        print("Please place sample videos in this directory and run the script again.")
        return None, None
    
    # List available videos
    videos = [f for f in os.listdir(sample_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    if not videos:
        print(f"No sample videos found in {sample_dir}")
        print("Please place some video files in this directory and run the script again.")
        return None, None
    
    print("\nAvailable sample videos:")
    for i, video in enumerate(videos):
        print(f"{i+1}. {video}")
    
    # Ask user to select a video
    while True:
        try:
            choice = int(input(f"Select a video (1-{len(videos)}): "))
            if 1 <= choice <= len(videos):
                selected = videos[choice-1]
                video_path = os.path.join(sample_dir, selected)
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    return cap, selected
                else:
                    print(f"Could not open {selected}. Try another one.")
            else:
                print(f"Please enter a number between 1 and {len(videos)}")
        except ValueError:
            print("Please enter a valid number")

def test_hand_gesture_detector():
    """Test the improved hand gesture detector with a webcam or video file"""
    print("Improved Hand Gesture Detector Test")
    print("==================================\n")
    
    # Get video source
    source_info = get_video_source()
    if source_info is None:
        print("No valid video source available. Exiting.")
        return
        
    cap, source_name = source_info
    
    # If no valid source is found
    if cap is None:
        print("No valid video source selected. Exiting.")
        return
    
    # Get webcam resolution
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Video resolution: {width}x{height}")
    
    # Initialize the detector
    detector = EnhancedHandGestureDetector()
    
    # For FPS calculation
    prev_time = time.time()
    fps_history = []
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            
            # Handle end of video file or loop
            if not ret:
                if source_name != "webcam":
                    # For video files, loop back to the beginning
                    print("Restarting video...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        print("Could not restart video. Exiting.")
                        break
                else:
                    print("Could not read frame from webcam. Exiting.")
                    break
            
            # Process frame with the detector
            controls, processed_frame, data_panel = detector.detect_gestures(frame)
            
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            fps_history.append(fps)
            if len(fps_history) > 30:  # Keep average over last 30 frames
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)
            prev_time = current_time
            
            # Add FPS to the detector for display
            detector.fps = avg_fps
            
            # Resize data panel if needed to match the height of the frame
            h, w = processed_frame.shape[:2]
            data_panel_resized = cv2.resize(data_panel, (data_panel.shape[1], h))
            
            # Create a combined display
            combined_display = np.hstack((processed_frame, data_panel_resized))
            
            # Show the combined frame
            cv2.imshow("Hand Gesture Detection", combined_display)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("Test finished.")

if __name__ == "__main__":
    print("Hand Gesture Detection Test with Fixed UI")
    print("Options:")
    print("1. Test with improved detector")
    print("2. List available cameras")
    print("3. Exit")
    
    choice = input("Select option (1-3): ")
    
    if choice == "1":
        test_hand_gesture_detector()
    elif choice == "2":
        list_available_cameras()
    else:
        print("Exiting.")
