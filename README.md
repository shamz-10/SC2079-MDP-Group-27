# SC2079 MDP Group 27

## Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Project Components](#project-components)
  - [Raspberry Pi](#raspberry-pi)
  - [PC Client](#pc-client)
  - [Algorithm](#algorithm)
  - [Image Recognition](#image-recognition)
  - [Android Application](#android-application)
  - [STM](#stm)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Team](#team)

## Overview

This project implements an autonomous robot system capable of:
- **Task 1**: Automatic movement and image recognition task
- **Task 2**: Fastest car task using visual recognition

The system integrates multiple components including a Raspberry Pi controller, PC-based image recognition, Android application for control/monitoring, and STM32 microcontroller for motor control.

## 🏗️ System Architecture

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   Android   │◄───────►│ Raspberry Pi │◄───────►│  PC Client  │
│ Application │  BT     │   (Main)     │  WiFi   │  (Vision)   │
└─────────────┘         └──────┬───────┘         └─────────────┘
                               │ Serial
                        ┌──────▼───────┐
                        │    STM32     │
                        │              │
                        └──────────────┘
```

## 📂 Project Structure

```
SC2079-MDP-Group-27/
├── Algorithm/                  # Path planning and navigation
│   ├── task1_manager.py
│   ├── algo_testing.py
│   ├── algo_backwards.py
│   ├── socket_trace_server.py
│   ├── movement_trace.json     # Generated path output
│   ├── obstacle_visit_order.json
│   └── obstacles.json          # Obstacle configuration
├── STM/          # STM32 MCU
│   ├── Src/
│   │   ├── main.c # Main file
│   ├── Inc/
│   ├── MDP task2.ioc
│   ├── MDP task1.ioc
├── image_recognition/          # Computer vision module
│   ├── model_inference.py
│   ├── stitch_images.py
│   └── models/                 # YOLO model weights
│       ├── v3.pt
│       └── task2.pt
├── mdp-rpi/                    # Raspberry Pi code
│   ├── main.py
│   ├── android.py
│   ├── pc.py
│   ├── stm.py
│   ├── camera.py
│   └── rpi_config.py
├── captured_images/            # Captured image storage
├── images_result/              # Processed images
├── PC_client.py               # PC client main file
└── README.md
```

## Project Components

### Raspberry Pi

**Location**: `/mdp-rpi/`

The Raspberry Pi serves as the central coordinator for the robot, managing communication between all components.

#### Key Files:
- `main.py` - Main entry point and thread manager
- `android.py` - Bluetooth interface for Android communication
- `pc.py` - WiFi socket interface for PC communication
- `stm.py` - Serial interface for STM32 communication
- `camera.py` - Camera control for image capture
- `rpi_config.py` - Configuration settings

#### Features:
- Multi-threaded message routing between components
- Image capture and transmission to PC
- Command forwarding between Android and STM32

### PC Client

**Location**: `/PC_client.py`

The PC Client handles computationally intensive tasks including image recognition and algorithm execution.

#### Key Features:
- **Image Recognition**: Uses YOLOv8 model for object detection
- **Path Planning**: Integrates with algorithm module for navigation
- **Image Processing**: Handles image stitching for Task 2
- **Socket Communication**: Receives images and commands from Raspberry Pi
- **Retry Mechanism**: Implements retry logic for failed image recognition

### Algorithm

**Location**: `/Algorithm/`

The algorithm module handles path planning and navigation logic for the robot.

#### Key Files:
- `task1_manager.py` - Main task management and path generation
- `algo_final.py` - Core pathfinding algorithm

#### Output Files:
- `movement_trace.json` - Generated movement commands and path
- `obstacle_visit_order.json` - Obstacle visiting sequence
- `obstacles.json` - Obstacle configuration input

#### Key Features:
- A* pathfinding with obstacle avoidance
- Command segmentation for image recognition points
- Optimal path calculation for obstacle visitation
- Support for custom obstacle configurations

### Image Recognition

**Location**: `/image_recognition/`

Computer vision module using YOLOv8 for image symbol recognition.

#### Key Files:
- `model_inference.py` - YOLO model inference logic
- `stitch_images.py` - Image stitching for Task 2
- `models/` - Trained YOLO model weights
  - `v3.pt` - Task 1 model
  - `task2.pt` - Task 2 model

### Android Application

**Location**: *Code not yet pushed*

The Android application module provides the graphical user interface and Bluetooth communication layer for controlling, visualizing, and monitoring the robot in real time.

#### Key Files:
- `HomeFragment.java`  – Main control interface handling robot updates, obstacle placement, and task initiation
- `RpiController.java` – Handles JSON communication and message parsing between the app and Raspberry Pi
- `Map.java` – Custom grid view for rendering robot, obstacles, and paths
- `RecyclerAdapter.java` – Manages obstacle list items and visibility within the UI

#### Key Features:
- 20×20 grid-based visualization with real-time updates
- Dialog-based obstacle placement via manual input (x, y, direction)
- Automatic parsing and animation of received path data
- Two-way Bluetooth communication with structured JSON handling


### STM

**Location**: `/STM/Src/main.c`

The STM acts as the car's control layer to read sensors, drive motors/steering while exposing fixed motion movesets to the Raspberry Pi for safe repeatable movement.

#### Key Files:
- `main.c` - Main code
- `MDP task1.ioc` - IOC file for task1
- `MDP task2.ioc` - IOC file for task2
- 
#### Key Features:
- Sensor fusion for yaw (gyroscope + magnetometer) with angle unwrapping and drift correction for precise turns/heading hold.
- Real-time motion control on STM32F407: Pulse-Width Modulation (PWM) for DC motors & servo steering with encoder feedback.
- Closed-loop stopping to target using encoder + ultrasonic/IR guard bands to avoid overshoot into obstacles.
- Reliable motion primitives: drive distance (cm), turn (°), arc, soft/hard brake—each with tunable PID/PD gains and speed profiles.
- FreeRTOS architecture: dedicated tasks for control, sensors, and comms with non-blocking software timers; on-device calibration & OLED debug.

## 🛠 Setup and Installation

### Prerequisites

#### Raspberry Pi:
```bash
# Install Python dependencies
pip3 install pyserial
pip3 install pybluez
```

#### PC Client:
```bash
# Install Python 3.8+
pip install ultralytics
```

### Hardware Setup

1. **Raspberry Pi Configuration**:
   - Enable Serial port in `raspi-config`
   - Enable Bluetooth
   - Configure WiFi network

2. **Network Configuration**:
   - Ensure Raspberry Pi and PC are on the same network
   - Update IP addresses in configuration files

3. **STM32 Connection**:
   - Connect STM32 to Raspberry Pi via USB serial
   - Note the serial port (usually `/dev/ttyUSB0`)

## Usage

### Running the System

#### 1. Start PC Client:
```bash
cd /path/to/SC2079-MDP-Group-27
python PC_client.py
```

#### 2. Start Raspberry Pi:
```bash
cd mdp-rpi
python3 main.py
```

#### 3. Connect Android App:
- Launch Android application
- Enable Bluetooth
- Connect to Raspberry Pi (device name: "RPi_MDP")

#### 4. Configure Obstacles (Task 1):
- Use Android app to place obstacles on grid
- Send configuration to robot

#### 5. Start Task:
- Select NAVIGATION (Task 1) or FASTEST_PATH (Task 2)
- Press "Start" button

## Team ✨
<table>
  <tr>
    <td align="center"><a href="https://github.com/shamz-10"><img src="https://avatars.githubusercontent.com/shamz-10" width="100px;" alt=""/><br /><sub><b>Shammas</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/mokshittjain"><img src="https://avatars.githubusercontent.com/mokshittjain" width="100px;" alt=""/><br /><sub><b>Mokshit</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/sreeisme"><img src="https://avatars.githubusercontent.com/sreeisme" width="100px;" alt=""/><br /><sub><b>Sree</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/ayyyman22"><img src="https://avatars.githubusercontent.com/ayyyman22" width="100px;" alt=""/><br /><sub><b>Ayman</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/CradleStrife"><img src="https://avatars.githubusercontent.com/CradleStrife" width="100px;" alt=""/><br /><sub><b>Lai Yi</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/JeremyCEY"><img src="https://avatars.githubusercontent.com/JeremyCEY" width="100px;" alt=""/><br /><sub><b>En Yao</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/27July"><img src="https://avatars.githubusercontent.com/27July" width="100px;" alt=""/><br /><sub><b>Zi Hao</b></sub></a><br /></td>
  </tr>
</table>

---
