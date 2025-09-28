#!/bin/bash

# Setup script for Face Detection & Motion Sensor Application
# This script fixes the libcamera dependency issue for PiCamera2

echo "Setting up PiCamera2 dependencies for Face Detection & Motion Sensor..."

# Check if running on Raspberry Pi
if grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    echo "Raspberry Pi detected. Installing libcamera dependencies..."
    
    # Update package list
    sudo apt update
    
    # Install libcamera and related packages
    echo "Installing libcamera system libraries..."
    sudo apt install -y libcamera-dev libcamera-tools python3-libcamera
    
    # Install additional dependencies
    echo "Installing additional camera dependencies..."
    sudo apt install -y python3-picamera2 python3-pil python3-numpy
    
    # Install Python packages
    echo "Installing Python packages..."
    pip3 install picamera2 opencv-python flask pillow
    
    echo ""
    echo "PiCamera2 setup complete!"
    echo "You can now run: python3 main.py"
    
else
    echo "ERROR: This script is designed for Raspberry Pi systems."
    echo "PiCamera2 requires libcamera which is only available on Raspberry Pi."
    echo "Please run this script on your Raspberry Pi."
    exit 1
fi

echo ""
echo "Setup complete! PiCamera2 should now work properly."
