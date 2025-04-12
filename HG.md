# Hand Gesture Car Control System
## Software Requirements Specification

## Table of Contents
1. [Introduction](#1-introduction)
   1. [Purpose](#11-purpose)
   2. [Scope](#12-scope)
   3. [Definitions and Acronyms](#13-definitions-and-acronyms)
2. [Overall Description](#2-overall-description)
   1. [Product Perspective](#21-product-perspective)
   2. [Product Functions](#22-product-functions)
   3. [User Characteristics](#23-user-characteristics)
   4. [Operating Environment](#24-operating-environment)
   5. [Design and Implementation Constraints](#25-design-and-implementation-constraints)
3. [System Architecture](#3-system-architecture)
   1. [Component Diagram](#31-component-diagram)
   2. [Directory Structure](#32-directory-structure)
4. [Functional Requirements](#4-functional-requirements)
   1. [Hand Gesture Detection](#41-hand-gesture-detection)
   2. [Car Control](#42-car-control)
   3. [Game Interface](#43-game-interface)
   4. [Sound System](#44-sound-system)
5. [Non-Functional Requirements](#5-non-functional-requirements)
   1. [Performance](#51-performance)
   2. [Usability](#52-usability)
   3. [Reliability](#53-reliability)
   4. [Portability](#54-portability)
6. [External Interface Requirements](#6-external-interface-requirements)
   1. [User Interfaces](#61-user-interfaces)
   2. [Hardware Interfaces](#62-hardware-interfaces)
   3. [Software Interfaces](#63-software-interfaces)
7. [Other Requirements](#7-other-requirements)
   1. [Troubleshooting](#71-troubleshooting)
   2. [Analytics](#72-analytics)

## 1. Introduction

### 1.1 Purpose
This Software Requirements Specification (SRS) document describes the requirements for the Hand Gesture Car Control System, a Python application that allows users to control a virtual car using hand gestures captured through a webcam.

### 1.2 Scope
The Hand Gesture Car Control System is an interactive application that uses computer vision and machine learning to detect hand movements and translate them into car controls in a virtual environment. The system includes:

- Real-time hand gesture detection using MediaPipe
- A virtual car simulation with physics-based movement
- Game elements including obstacles and scoring
- Sound effects that respond to actions
- Multiple difficulty modes
- Camera selection interface

The application can be used for entertainment, hand-eye coordination training, or as a tech demonstration for hand gesture recognition.

### 1.3 Definitions and Acronyms

| Term | Definition |
|------|------------|
| CV | Computer Vision |
| ML | Machine Learning |
| FPS | Frames Per Second |
| GUI | Graphical User Interface |
| UDP | User Datagram Protocol |
| API | Application Programming Interface |
| MCP | Metacarpophalangeal (knuckle) |

## 2. Overall Description

### 2.1 Product Perspective
The Hand Gesture Car Control System is a standalone application but is designed with a modular architecture that separates the hand detection, car control, and game interface components. This design allows for future expansion, such as connecting to a physical car or integrating with other systems.

The system uses the webcam to capture hand movements, processes these movements to extract control data, and applies these controls to a virtual car in a game environment. The game includes obstacles, scoring, and different difficulty modes.

### 2.2 Product Functions
The main functions of the system include:

- **Hand tracking and gesture recognition**: Detect hand landmarks and recognize gestures
- **Control translation**: Convert detected gestures into control signals
- **Virtual car simulation**: A physics-based car model that responds to controls
- **Obstacle generation**: Create and manage road obstacles
- **Sound effects**: Generate and play appropriate sound effects
- **Game mechanics**: Scoring, collisions, and game modes
- **User interface**: Menus, status displays, and control settings

### 2.3 User Characteristics
The intended users of this system include:

- Casual users interested in gesture-based interfaces
- Gaming enthusiasts
- Students or professionals learning about computer vision and gesture recognition
- Demonstrators showcasing computer vision technology

No special technical expertise is required to use the application, though users should be comfortable with basic computer operations and have normal hand mobility.

### 2.4 Operating Environment
The application runs on standard desktop or laptop computers with the following requirements:

- Operating System: Windows, macOS, or Linux
- Minimum 2 GB RAM (4 GB recommended)
- Webcam with minimum 640x480 resolution
- Processor: Dual-core 2 GHz or better
- Python 3.7 or higher with required packages

### 2.5 Design and Implementation Constraints
The following constraints apply to the system design and implementation:

- Real-time processing requirements for hand detection
- Limited by webcam quality and lighting conditions
- Frame rate must remain above 30 FPS for responsive controls
- Dependency on third-party libraries (OpenCV, MediaPipe, PyGame)
- Python language performance limitations

## 3. System Architecture

### 3.1 Component Diagram
The system consists of the following main components:

1. **Hand Detection Module**: Handles webcam input and detects hand gestures
2. **Car Control Module**: Translates gestures into car control commands
3. **Game Engine**: Manages the game environment, obstacles, and physics
4. **Sound Manager**: Controls audio feedback
5. **User Interface**: Provides visual feedback and menu systems

These components interact in a pipeline: hand gestures are detected, translated to controls, applied to the car, and the game state is updated and rendered with appropriate sounds.

### 3.2 Directory Structure
The application follows a modular directory structure:

```
hand_gesture_car_control/
├── README.md
├── requirements.txt
├── main.py
├── config.py
├── run.py
├── car_control.py
├── main_menu.py
├── debug_utils.py
├── troubleshoot.py
├── hand_detector/
│   ├── __init__.py
│   ├── detector.py
│   ├── gestures.py
│   ├── improved_hand_gesture_detector.py
│   └── tracking.py
├── game/
│   ├── __init__.py
│   ├── car.py
│   ├── objects.py
│   ├── renderer.py
│   ├── physics.py
│   ├── audio_manager.py
│   ├── camera_manager.py
│   └── start_game.py
├── utils/
│   ├── __init__.py
│   ├── camera.py
│   ├── sound.py
│   └── ui.py
├── sounds/
│   ├── engine_idle.wav
│   └── engine_revving.wav
└── assets/
    ├── car.png
    ├── road.png
    └── obstacles/
        ├── obstacle1.png
        ├── obstacle2.png
        └── obstacle3.png
```

## 4. Functional Requirements

### 4.1 Hand Gesture Detection

#### 4.1.1 Hand Landmark Detection
- The system shall detect hand landmarks using MediaPipe's Hand solution
- The system shall track up to one hand in the frame at a time
- The system shall identify and track at least 21 hand landmarks with an accuracy of 90% or higher
- The system shall visualize hand landmarks and connections on the detection frame

#### 4.1.2 Gesture Recognition
- The system shall recognize the following gestures:
  - Steering (hand tilt for left/right)
  - Throttle (hand height for speed)
  - Brake (fist gesture)
  - Boost (thumb up, other fingers curled)
  - Stop (open palm)
- The system shall implement gesture smoothing to reduce jitter
- The system shall maintain a detection rate of at least 15 frames per second
- The system shall filter unstable gestures with a stability threshold

#### 4.1.3 Camera Management
- The system shall detect available webcams
- The system shall allow the user to select a webcam from available options
- The system shall handle camera disconnection or errors gracefully
- The system shall initialize the selected camera with appropriate resolution settings

### 4.2 Car Control

#### 4.2.1 Control Mapping
- The system shall map recognized gestures to car controls:
  - Hand tilt (rotation) → Steering
  - Hand position (height) → Throttle
  - Fist → Brake
  - Thumb up → Boost
  - Open palm → Emergency stop
- The system shall apply smoothing to control inputs to prevent jerky car movement
- The system shall handle the absence of hand detection by gradually slowing the car

#### 4.2.2 Car Physics
- The system shall implement a physics model for car movement including:
  - Acceleration and deceleration
  - Steering with speed-dependent handling
  - Collision detection with obstacles
  - Boundary constraints
- The car shall respond differently at different speeds
- The car shall visually indicate braking and boosting states

#### 4.2.3 Command Processing
- The system shall maintain a command queue for car control
- The system shall implement command throttling to prevent redundant commands
- The system shall track command success/failure rate
- When in simulation mode, the system shall visualize commands without sending them

### 4.3 Game Interface

#### 4.3.1 Game Modes
- The system shall provide at least 5 different game modes:
  - Practice Mode: No obstacles
  - Easy Mode: Few obstacles at slow pace
  - Normal Mode: Standard gameplay
  - Hard Mode: Many fast obstacles
  - Time Trial: Race against the clock
- Each mode shall have configurable settings for:
  - Obstacle frequency
  - Obstacle speed
  - Score multiplier
  - Time limit (optional)

#### 4.3.2 Gameplay Mechanics
- The system shall generate random obstacles on the road
- The system shall track score based on obstacles passed
- The system shall detect and handle collisions
- The system shall provide visual feedback for collisions
- The system shall support a pause function
- The system shall display game over screen when appropriate

#### 4.3.3 User Interface
- The system shall display current score, speed, and time
- The system shall visualize the car's position and movement
- The system shall provide visual indicators for game state
- The system shall include a mute button for sound control
- The system shall display appropriate loading and error messages

### 4.4 Sound System

#### 4.4.1 Sound Effects
- The system shall provide sound effects for:
  - Engine idle
  - Engine revving
  - Engine boost
  - Collisions
  - Power-ups
  - Braking
  - Game over
- The system shall synthesize sounds using wave generation
- The system shall adjust engine sound based on car speed

#### 4.4.2 Sound Management
- The system shall support muting/unmuting of all sounds
- The system shall adjust volume levels appropriately
- The system shall handle sound errors gracefully
- The system shall clean up sound resources when they are no longer needed

## 5. Non-Functional Requirements

### 5.1 Performance

#### 5.1.1 Response Time
- The system shall have a maximum end-to-end latency of 200ms from gesture to car movement
- The system shall maintain a minimum frame rate of 30 FPS for the game visualization
- The system shall process hand detection at a minimum of 15 FPS

#### 5.1.2 Resource Usage
- The system shall use no more than a maximum of 2 GB of RAM
- The system shall use no more than 50% of CPU on a modern dual-core processor
- The system shall function smoothly on integrated graphics cards

### 5.2 Usability

#### 5.2.1 User Interface
- The system shall provide clear visual feedback for all user actions
- The system shall include a main menu with game mode selection
- The system shall display instructions for gesture controls
- The system shall provide intuitive visualization of hand tracking

#### 5.2.2 Learning Curve
- New users shall be able to learn basic controls within 2 minutes
- The system shall include a practice mode for learning controls
- The system shall provide visual aids for gesture recognition

### 5.3 Reliability

#### 5.3.1 Error Handling
- The system shall handle webcam errors gracefully
- The system shall recover from frame processing errors
- The system shall implement automatic timeout for lost hand tracking
- The system shall provide appropriate error messages

#### 5.3.2 Stability
- The system shall run continuously for at least 1 hour without crashes
- The system shall properly release resources when closed
- The system shall implement command stability tracking to prevent jitter

### 5.4 Portability

#### 5.4.1 Platform Compatibility
- The system shall be compatible with Windows 10/11, macOS, and major Linux distributions
- The system shall handle different screen resolutions
- The system shall adapt to different camera specifications

## 6. External Interface Requirements

### 6.1 User Interfaces
- The system shall provide a main menu screen for game mode selection
- The system shall provide a camera selection interface
- The system shall display hand tracking visualization
- The system shall include a game screen with car, road, and obstacles
- The system shall display a pause menu when paused
- The system shall present a game over screen at the end of a game

### 6.2 Hardware Interfaces
- The system shall interface with standard webcams via the OpenCV API
- The system shall support a minimum webcam resolution of 640x480
- The system shall work with both integrated and external webcams
- The system shall use the GPU for MediaPipe processing if available

### 6.3 Software Interfaces
- The system shall use MediaPipe for hand detection
- The system shall use OpenCV for image processing
- The system shall use PyGame for game rendering
- The system shall use NumPy for numerical operations
- The system shall interface with the operating system for socket communication (when controlling physical car)

## 7. Other Requirements

### 7.1 Troubleshooting
- The system shall include a connectivity troubleshooting utility
- The system shall log errors and debug information to console
- The system shall detect and report network configuration issues
- The system shall provide recommendations for resolving common issues

### 7.2 Analytics
- The system shall include a reaction time analyzer for performance measurement
- The system shall track and analyze component performance
- The system shall visualize performance metrics
- The system shall identify performance bottlenecks