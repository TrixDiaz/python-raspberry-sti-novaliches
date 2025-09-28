# Face Detection & Motion Sensor with Flask Streaming

A comprehensive Python application that combines face detection, motion sensing, and real-time camera streaming using OpenCV, PiCamera2, and Flask.

## Features

- **Face Detection**: Uses Haar cascade classifier to detect faces and draws green bounding boxes around them
- **Motion Detection**: Detects motion in the camera feed and displays "Motion" text in the upper right corner
- **Real-time Streaming**: Streams the camera feed via Flask web server accessible through IP address
- **Web Interface**: Clean, responsive web interface for viewing the camera feed

## Requirements

- Raspberry Pi with camera module
- Python 3.7+
- OpenCV
- PiCamera2
- Flask

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure the Haar cascade model is in the correct location**:
   - The `haarcascade_frontalface_default.xml` file should be in the `model/` directory
   - This file is included in the project

## Usage

1. **Run the application**:

   ```bash
   python main.py
   ```

2. **Access the web interface**:

   - Local access: `http://localhost:5000`
   - Network access: `http://[YOUR_PI_IP_ADDRESS]:5000`

3. **Features**:
   - **Face Detection**: Green rectangles will appear around detected faces
   - **Motion Detection**: Red "Motion" text will appear in the upper right corner when motion is detected
   - **Real-time Streaming**: Live camera feed with all detections overlaid

## API Endpoints

- `/` - Main web interface
- `/video_feed` - Video stream endpoint
- `/status` - JSON API to check motion detection status

## Configuration

The application can be customized by modifying the following parameters in `main.py`:

- **Face Detection**:

  - `scaleFactor`: Controls detection sensitivity (default: 1.1)
  - `minNeighbors`: Minimum neighbors for detection (default: 5)
  - `minSize`: Minimum face size (default: 30x30)

- **Motion Detection**:

  - Motion threshold: `cv2.contourArea(contour) > 500`
  - Display duration: `motion_display_duration = 3` seconds

- **Camera Settings**:
  - Resolution: 640x480 (main), 320x240 (lores)
  - Format: RGB888

## Troubleshooting

1. **Camera not detected**: Ensure the camera is properly connected and enabled
2. **Permission errors**: Run with appropriate permissions or use `sudo`
3. **Performance issues**: Reduce camera resolution or frame rate
4. **Network access**: Ensure firewall allows port 5000

### Installing dlib on Raspberry Pi (optional)

If `pip install -r requirements.txt` fails trying to build `dlib` from source, you can either rely on a prebuilt wheel (piwheels) or install the system build-dependencies and build from source:

- Prefer piwheels (default on Raspberry Pi OS) — pip will usually fetch a prebuilt aarch64 wheel:

  ```bash
  pip install dlib
  ```

- If a wheel is not available and pip tries to build from source, install these packages first:

  ```bash
  sudo apt update
  sudo apt install -y build-essential cmake libopenblas-dev liblapack-dev libjpeg-dev libpng-dev python3-dev
  ```

- Then retry:

  ```bash
  pip install dlib
  ```

If building still fails, capture the pip error log and open an issue with the error output.

## Technical Details

- **Face Detection**: Uses OpenCV's Haar cascade classifier
- **Motion Detection**: Uses MOG2 background subtractor
- **Streaming**: Flask with multipart response for MJPEG streaming
- **Threading**: Separate thread for frame processing to maintain performance

## License

This project is open source and available under the MIT License.
