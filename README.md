# 🎮 Hand Gesture Vehicle Control System - Hand Gesture Control

## 📁 Project Structure
```
Hand_mov_Z/
├── 📝 debug_logs/           # Debug logs
├── 🌐 env/                 # Virtual environment (Python packages)
├── 🔊 sounds/              # Sound files (optional - creates procedural sounds if not present)
│
├── 🚀 main.py              # Main entry point (includes menu)
├── 🏃 run.py               # Alternative entry point
├── 🏋️ training_mode.py     # Training mode
├── 📋 requirements.txt     # Dependencies list
├── ⚙️ config.py            # System configuration
├── 🐛 debug_config.py      # Debug configuration
└── 🧪 test_game_sound.py   # Sound system test
```

## 📋 Project Description
Final project for Full Stack Python course - An innovative system that allows controlling a car game using real-time hand gestures. The system uses a webcam to detect hand movements and translates them into car control commands in the game.

## 🎯 Project Goal
Creating a natural and intuitive user interface for computer games, using computer vision and machine learning technologies, as part of the graduation project in the Full Stack Python course.

## 🛠️ Technologies

### Backend & Core Logic
- **Python 3.8+** - Main programming language (recommended 3.8-3.12)
- **OpenCV** - Image processing and video capture from camera
- **MediaPipe** - Hand detection and tracking of landmarks
- **NumPy** - Mathematical calculations and array processing

### Frontend & UI
- **Pygame** - Game engine and user interface
- **OpenCV UI** - Additional windows for data display

### Full Stack Python Features
- **Modular Architecture** - Separation between logic, view, and control
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
│   ├── improved_hand_gesture_detector.py  # Improved version
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

## 🚀 Launch Methods

The system offers two different launch methods:

### 📋 `python main.py` - Launch with full menu
```bash
python main.py
```
**What it does:**
- Opens interactive main menu
- Allows difficulty selection (Easy/Normal/Hard)
- Camera and debug settings
- Detailed usage instructions
- **Recommended for new users**

### ⚡ `python run.py` - Quick launch
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
| 👍 Thumbs up | Maximum acceleration | Thumb up, other fingers closed |
| Hand tilt | Steering | Tilt left/right for turning |
| ↕️ Hand height | Speed | Up = fast, down = slow |

### Advanced Gestures
- **Thumb angle** - Precise steering control (0° = straight, ±90° = full turn)
- **Finger distance** - Command intensity

## 🚗 Vehicle Physics

### Realistic Physics System
- **Accurate directional movement** - Vehicle moves exactly in the direction it's facing
- **Speed effect on turning** - At high speeds, turns are less sharp
- **Centrifugal force** - Vehicle is pushed outward in turns
- **Automatic center return** - When driving straight, vehicle returns to road center

## 🎯 Game Modes

### 1. Practice Mode
- No obstacles
- No time limit
- Score multiplier: x0.5
- Ideal for learning controls

### 2. Easy Mode
- Few and slow obstacles
- No time limit
- Score multiplier: x1.0

### 3. Normal Mode
- Standard balance
- No time limit
- Score multiplier: x1.5

### 4. Hard Mode
- Many fast obstacles
- No time limit
- Score multiplier: x2.0

### 5. Time Trial - Race against time
- 2 minutes only
- Fast obstacles
- Score multiplier: x2.5

## 💻 System Requirements

### Hardware
- **Webcam** (720p and above recommended)
- **Processor:** Intel i3 / AMD Ryzen 3 and above (i5/Ryzen 5 recommended)
- **RAM Memory:** 4GB minimum (8GB recommended)
- **Graphics card** supporting OpenGL

### Software
- **Operating System:** Windows 10/11, macOS 10.14+, or Linux Ubuntu 18.04+
- **Python** 3.8-3.12 (tested with Python 3.8, 3.9, 3.10, 3.11, 3.12)
- **pip** for package management
- **Camera** with updated drivers

## 🔧 Installation and Setup

### 1. Check system requirements
```bash
python --version  # Ensure Python version is 3.8+
pip --version     # Ensure pip is installed
```

### 2. Clone the project
```bash
git clone https://github.com/sharon-bd/Hand_mov_Z.git
cd Hand_mov_Z
```

### 3. Create virtual environment
```bash
python -m venv env

# Windows:
env\Scripts\activate

# Mac/Linux:
source env/bin/activate
```

### 4. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. System check
```bash
# Sound system test
python test_game_sound.py

# Hand gesture connection test (optional)
python hand_detector/connection_test.py
```

### 6. Launch the game

**Regular launch (recommended):**
```bash
python main.py
```

**Quick launch:**
```bash
python run.py
```

## 🎮 Game Instructions

### Keyboard controls (backup)
- **Arrows / WASD** - Movement
- **Space** - Acceleration
- **ESC** - Pause/Exit
- **P** - Pause (automatically mutes sound)
- **M** - Mute/Unmute
- **H** - Help
- **F1** - Debug mode

### Game Tips
- **Initial calibration** - Ensure hand is fully visible in camera
- **Good lighting** - Important for even lighting without shadows
- **Clean background** - Uniform background improves detection
- **Decisive movements** - Clear and sharp movements give better results

## 🔊 Dynamic Sound System

### Procedural Sound (creates sounds in real-time)
The system **does not require external sound files** - it creates all sounds in code!

**How it works:**
- If there are no .wav files in the `sounds/` folder - the system creates artificial sounds
- Sounds are created using sine and cosine waves at different frequencies
- Each sound is adapted to the specific action

### Types of sounds created:
- **Engine sound** - Changes according to speed (frequency 80-200 Hz)
- **Boost effects** - Rising frequency sound (200-800 Hz)
- **Collisions** - White noise with fast decay
- **Menu sounds** - Short tones for clicks
- **3D system** - Sounds change according to position in game

### Procedural sound advantages:
✅ **No dependency on external files** - Game works immediately without media files  
✅ **Small file size** - No need to store large .wav files  
✅ **Dynamic adaptation** - Sounds change according to game state  
✅ **Consistent quality** - Not dependent on external recording quality

## 🐛 Common Troubleshooting

### Installation errors
```bash
# If there's a problem with MediaPipe
pip install --upgrade mediapipe

# If there's a problem with OpenCV
pip install opencv-python

# If there's a problem with Pygame
pip install pygame

# Reinstall all packages
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

### Camera not found
- Check that camera is connected and working
- Close other programs that might be using the camera
- Try running in simulation mode (without camera)
- Check camera permissions in operating system
- **Windows:** Check in Settings > Privacy > Camera
- **Mac:** Check in System Preferences > Security & Privacy > Camera

### Low performance
- Lower camera resolution in `config.py`
- Close other resource-consuming programs
- Use windowed mode instead of fullscreen
- Reduce graphics quality in game

### Inaccurate gesture detection
- Improve lighting - natural light or white LED
- Move hand away from camera (30-60 cm)
- Use uniform and bright background
- Ensure hand is in center of screen
- Clean camera lens

### Common Python errors
```bash
# If "Module not found" error appears
pip install [missing module name]

# If there's a problem with Python version
python -m pip install --upgrade setuptools wheel
```

## 🎯 Advanced Features

### 1. Advanced Physics System
- Centrifugal force calculations
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
- Smooth 60 FPS animations
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
- Gesture detection visualization

## 🔍 Debug Modes and Debugging

### 🎮 Game Debug Mode
**How to activate:** Press `D` during game
- **Location:** Bottom part of game screen
- **Display:** White text on transparent background
- **Information displayed:**
  - Current game FPS
  - Vehicle position (world and screen)
  - Vehicle speed and steering control
  - Rotation and movement direction
  - Number of obstacles on screen
  - Game state and remaining time

### 📊 Additional information displayed in terminal
During game, messages are displayed in terminal:
- `🔊 Engine sound playing at speed X.XX` - Engine sound
- `🤏 Gesture detected: [gesture type]` - Gesture detection
- `🎯 Steering Debug: Angle=X°, Steering=X.XX` - Steering information
- `🖐️ Palm Detection Debug` - Palm detection
- `💥 Turtle collision detected!` - Collision
- `🐢 Normal: Spawning turtle` - Obstacle creation

### 🎛️ Additional keys for developers
- **P** - Pause
- **M** - Mute sound
- **H** - Help
- **ESC** - Return to menu or exit

## 🔄 Recent Updates

### Current Version
- Improved compatibility for different Python versions
- Performance improvements in gesture detection system
- Fixed compatibility issues with different operating systems
- Improved error system and user messages

### Planned Future Features
- Support for two-hand gesture detection
- Local multiplayer mode
- Save personal settings
- Personal record tracking

## 📄 License

This project was created as part of the Full Stack Python course and submitted as a graduation project.
All rights reserved to student Sharon Ben-Dror.

## 👨‍💻 Development Team

This project was developed as part of the Full Stack Python course by:
**Sharon Ben-Dror** - JohnBryce Academy

---

## 📞 Contact

For questions, suggestions, or bug reports:
- GitHub Issues: [Report an issue](https://github.com/sharon-bd/Hand_mov_Z/issues)
- GitHub Repository: [Hand_mov_Z](https://github.com/sharon-bd/Hand_mov_Z)

---

⭐ If you liked the project, don't forget to give it a star on GitHub!
