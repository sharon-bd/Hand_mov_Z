# 🎮 Hand Gesture Car Control System

---
**Project Summary:**
This repository is an educational project, developed as a final assignment for the Full Stack Python course. It demonstrates advanced concepts in computer vision, real-time game control, and Python full stack development.
---

## 📂 File Structure
```
Hand_mov_Z/
├── 📝 debug_logs/               # Debug logs
├── 🌐 env/                      # Virtual environment (Python packages)
│
├── 🚀 main.py                    # Main entry point with menu
├── 🏃 run.py                     # Alternative entry point
├── 🏋️ training_mode.py          # Training mode
├── 📋 requirements.txt          # Dependencies list
├── ⚙️ config.py                 # System configuration
├── 🐛 debug_config.py           # Debug configuration
└── 🧪 test_game_sound.py        # Sound system test
```

## 📋 Project Description
A final project for the Full Stack Python course - an innovative system that allows controlling a car game using real-time hand gestures. The system uses a webcam to detect hand movements and translates them into control commands for the car in the game.

## 🎯 Project Goal
Creating a natural and intuitive user interface for computer games, using computer vision and machine learning technologies, as part of the graduation project in the Full Stack Python course.

## 🛠️ Technologies

### Backend & Core Logic
- **Python 3.12+** - Main programming language
- **OpenCV 4.11** - Image processing and video reading from camera
- **MediaPipe 0.10.21** - Hand detection and tracking of landmarks
- **NumPy 1.26** - Mathematical calculations and array processing

### Frontend & UI
- **Pygame 2.6.1** - Game engine and user interface
- **OpenCV UI** - Additional windows for data display

### Full Stack Python Features
- **MVC Architecture** - Separation between logic, view, and control
- **Multi-threading** - Parallel processing for camera and game
- **Real-time Processing** - Real-time data processing
- **Modular Design** - Modular structure with separate packages

## 📂 Project Structure
```
Hand_mov_Z/
│
├── 🎮 game/                      # Main game module
│   ├── __init__.py
│   ├── start_game.py            # Main game logic
│   ├── car.py                   # Car class with advanced physics
│   ├── moving_road.py           # Dynamic moving road system
│   ├── audio_manager.py         # Procedural sound manager
│   ├── camera_manager.py        # Camera management
│   ├── obstacle.py              # Game obstacles
│   ├── objects.py               # Additional game objects
│   ├── physics.py               # Physics engine
│   ├── renderer.py              # Rendering engine
│   └── road_generator.py        # Road generator
│
├── 🤚 hand_detector/             # Gesture detection module
│   ├── __init__.py
│   ├── connection.py            # Connection between gestures and control
│   ├── connection_test.py       # Gesture-control communication test
│   ├── improved_hand_gesture_detector.py  # Enhanced version
│   ├── simple_detector.py       # Simple gesture detector
│   ├── simple_hand_gesture_detector.py   # Simple version
│   ├── tracking.py              # Hand movement tracking
│   ├── main.py                  # Standalone module execution
│   └── run.py                   # Quick module execution
│
├── 📝 debug_logs/               # Debug logs
│
├── 🚀 main.py                    # Main entry point with menu
├── 🏃 run.py                     # Alternative entry point
├── 🏋️ training_mode.py          # Training mode
├── 📋 requirements.txt          # Dependencies list
├── ⚙️ config.py                 # System configuration
├── 🐛 debug_config.py           # Debug configuration
└── 🧪 test_game_sound.py        # Sound system test
```

## 🚀 Execution Methods

The system offers two different execution methods:

### 📋 `python main.py` - Full Menu Execution
```bash
python main.py
```
**What it does:**
- Opens an interactive main menu
- Allows difficulty level selection (Easy/Normal/Hard)
- Camera and debug settings
- Detailed usage instructions
- **Recommended for new users**

### ⚡ `python run.py` - Quick Execution
```bash
python run.py
```
**What it does:**
- Initializes the system directly
- Performs automatic dependency checks
- Launches the main menu (like main.py)
- **Recommended for developers and advanced users**

**Main difference:**
`run.py` includes additional system checks and is suitable for developers, while `main.py` is the standard entry point for end users.

## 🎮 Supported Hand Gestures

### Basic Gestures
| Gesture | Action | Description |
|---------|--------|-------------|
| 👋 Open palm (5 fingers) | Stop/Brake | Spread all fingers |
| 👍 Thumb up | Straight | Thumb pointing up |
| 👍↖️ Thumb up-left | Turn left | Thumb angled left |
| 👍↗️ Thumb up-right | Turn right | Thumb angled right |
| ↕️ Hand height | Speed | Up = fast, down = slow |

### Advanced Gestures
- **Thumb angle** - Precise steering control (0° = straight, ±90° = full turn)
- **Finger distance** - Command intensity

## 🚗 Car Physics

### Realistic Physics System
- **Precise directional movement** - The car moves exactly in the direction it's facing
- **Speed effect on turning** - At high speeds, turns are less sharp
- **Centrifugal force** - The car is pushed outward in turns
- **Automatic return to center** - When driving straight, the car returns to road center

## 🎯 Game Modes

### 1. Practice Mode
- No obstacles
- No time limit
- Score multiplier: x0.5
- Ideal for learning control

### 2. Easy Mode
- Few and slow obstacles
- 4 minutes time limit (240 seconds)
- Score multiplier: x0.8

### 3. Normal Mode
- Standard balance
- 3 minutes time limit (180 seconds)
- Score multiplier: x1.0

### 4. Hard Mode
- Many fast obstacles
- 2 minutes time limit (120 seconds)
- Score multiplier: x1.5

### 5. Time Trial
- 1 minute only (60 seconds)
- Fast obstacles
- Score multiplier: x2.0

## 💻 System Requirements

### Hardware
- **Webcam** (720p and above recommended)
- **Processor:** Intel i5 / AMD Ryzen 5 and above
- **RAM:** 8GB minimum
- **Graphics card** supporting OpenGL

### Software
- **Operating System:** Windows 10/11, macOS, or Linux
- **Python** 3.12 and above (tested with Python 3.12.5)
- **pip** for package management

## 🔧 Installation and Execution

### 1. Clone the Project
```bash
git clone https://github.com/sharon-bd/Hand_mov_Z.git
cd Hand_mov_Z
```

### 2. Create Virtual Environment
```bash
python -m venv env

# Windows:
env\Scripts\activate

# Mac/Linux:
source env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. System Check
```bash
python test_game_sound.py
```

### 5. Run the Game

**Regular execution (recommended):**
```bash
python main.py
```

**Quick execution:**
```bash
python run.py
```

## 🎮 Game Instructions

### Keyboard Controls (Backup)
- **Arrows / WASD** - Movement
- **Space** - Acceleration
- **ESC** - Pause/Exit
- **P** - Pause (automatically mutes sound)
- **M** - Mute/Unmute
- **H** - Help
- **F1** - Debug mode

### Game Tips
- **Initial calibration** - Make sure the hand is fully visible in the camera
- **Good lighting** - Even lighting without shadows is important
- **Clean background** - A uniform background improves detection
- **Decisive movements** - Clear and sharp movements give good results

## 🔊 Dynamic Sound System

### Procedural Sound
- **Engine sound** - Changes according to speed
- **Boost effects** - Frequency increase sound
- **Collisions** - White noise with decay
- **3D system** - Sounds change according to position

## 🐛 Troubleshooting

### Camera Not Found
- Check that the camera is connected and working
- Try running in simulation mode (without camera)
- Check camera permissions in the operating system

### Low Performance
- Lower camera resolution
- Close other programs
- Use windowed mode instead of fullscreen

### Inaccurate Gesture Recognition
- Improve lighting
- Distance the hand from the camera
- Use a uniform background

### Installation Problems
```bash
# Sound system check
python test_game_sound.py

# Hand gesture connection check
python hand_detector/connection_test.py
```

## 🏆 Advanced Features

### 1. Advanced Physics System
- Centrifugal force calculation
- Momentum in turns
- Realistic sliding

### 2. Multi-threading
- Separate thread for camera
- Separate thread for gesture processing
- Safe synchronization between threads

### 3. Performance Analysis
- Real-time FPS measurement
- Processing time monitoring
- Dynamic optimization

### 4. Advanced Debug System
- Detailed logs
- Performance tracking
- Data visualization

## 📊 Statistics and Tracking

The system tracks:
- Total game time
- Cumulative score
- Gesture accuracy
- Response times
- Obstacles avoided

## 🎨 Design and User Experience

### User Interface
- Intuitive main menu
- Clear visual indicators
- Immediate gesture feedback
- Smooth transitions between screens

### Graphics
- Smooth animations at 60 FPS
- Visual effects (boost, collision)
- Particle system
- Dynamic lighting

## 🔄 Development Modes

### Training Mode
```bash
python training_mode.py
```
- Gesture recognition model training
- Data collection
- System calibration

### Debug Mode
- Press **F1** in game
- Real-time performance information
- Gesture recognition visualization

## 🤝 Contributing to the Project

The project is open to contributions! Please:
1. Fork the project
2. Create a new branch for features
3. Commit with clear messages
4. Open Pull Request

## 📄 License

This project was created as part of the Full Stack Python course and is submitted as a graduation project.
All rights reserved to the course students.

## 👨‍💻 Development Team

This project was developed as part of the Full Stack Python course by:
**Sharon Ben-Dror** - JohnBryce Academy

---

## 📞 Contact

For questions, suggestions, or bug reports:
- GitHub Issues: [Report an Issue](https://github.com/sharon-bd/Hand_mov_Z/issues)
- GitHub Repository: [Hand_mov_Z](https://github.com/sharon-bd/Hand_mov_Z)

---

⭐ If you liked the project, don't forget to give it a star on GitHub!
