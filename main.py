import cv2
import numpy as np
import threading
import time
import base64
import os
from datetime import datetime
from flask import Flask, Response, render_template_string, jsonify, request
from picamera2 import Picamera2
from database import save_motion_detection

class FaceMotionDetector:
    def __init__(self):
        # Initialize PiCamera2
        self.camera = Picamera2()
        self.camera_config = self.camera.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            lores={"size": (320, 240), "format": "YUV420"}
        )
        self.camera.configure(self.camera_config)
        self.camera.start()
        
        # Load Haar cascade for face detection
        cascade_path = os.path.join("model", "haarcascade_frontalface_default.xml")
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Motion detection variables - High sensitivity settings
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=16,  # Lower threshold for more sensitivity
            history=500       # Longer history for better background learning
        )
        self.motion_detected = False
        self.motion_timer = 0
        self.motion_display_duration = 3  # seconds
        
        # Additional motion detection variables
        self.previous_frame = None
        self.motion_sensitivity = 30  # Lower = more sensitive
        self.min_motion_area = 200    # Minimum area to trigger motion
        
        # Frame for streaming
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Motion detection storage
        self.last_motion_time = 0
        self.motion_cooldown = 10  # seconds between motion detections
        self.captures_dir = "captures"
        if not os.path.exists(self.captures_dir):
            os.makedirs(self.captures_dir)
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def process_frames(self):
        """Main processing loop for face detection and motion detection"""
        while True:
            try:
                # Capture frame from PiCamera2
                frame = self.camera.capture_array()
                
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Process frame for face detection and motion detection
                processed_frame = self.detect_faces_and_motion(frame_bgr)
                
                # Update current frame for streaming
                with self.frame_lock:
                    self.current_frame = processed_frame
                    
            except Exception as e:
                print(f"Error processing frame: {e}")
                time.sleep(0.1)
    
    def detect_faces_and_motion(self, frame):
        """Detect faces and motion in the frame"""
        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        # Draw bounding boxes around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Motion detection with high sensitivity
        motion_mask = self.background_subtractor.apply(frame)
        
        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        
        # Additional frame difference method for extra sensitivity
        motion_detected = False
        if self.previous_frame is not None:
            # Convert current frame to grayscale
            gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_previous = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate frame difference
            frame_diff = cv2.absdiff(gray_current, gray_previous)
            _, thresh = cv2.threshold(frame_diff, self.motion_sensitivity, 255, cv2.THRESH_BINARY)
            
            # Find contours in frame difference
            diff_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Check frame difference for motion
            for contour in diff_contours:
                if cv2.contourArea(contour) > self.min_motion_area:
                    motion_detected = True
                    break
        
        # Update previous frame
        self.previous_frame = frame.copy()
        
        # Also check background subtractor method
        motion_contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check for significant motion with lower threshold for higher sensitivity
        total_motion_area = 0
        for contour in motion_contours:
            area = cv2.contourArea(contour)
            if area > self.min_motion_area:  # Use configurable threshold
                total_motion_area += area
                motion_detected = True
        
        # Additional check: if total motion area is significant, trigger
        if total_motion_area > 1000:  # Total area threshold for large movements
            motion_detected = True
        
        # Update motion status and capture photo if motion detected
        current_time = time.time()
        if motion_detected:
            self.motion_detected = True
            self.motion_timer = current_time
            
            # Capture photo and save to database if enough time has passed
            if current_time - self.last_motion_time > self.motion_cooldown:
                self.capture_motion_photo(frame, total_motion_area)
                self.last_motion_time = current_time
        elif current_time - self.motion_timer > self.motion_display_duration:
            self.motion_detected = False
        
        # Display motion text in upper right corner
        if self.motion_detected:
            cv2.putText(frame, 'Motion', (frame.shape[1] - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame
    
    def adjust_sensitivity(self, sensitivity_level):
        """Adjust motion detection sensitivity
        Args:
            sensitivity_level: 'low', 'medium', 'high', 'ultra'
        """
        if sensitivity_level == 'low':
            self.motion_sensitivity = 50
            self.min_motion_area = 500
        elif sensitivity_level == 'medium':
            self.motion_sensitivity = 30
            self.min_motion_area = 300
        elif sensitivity_level == 'high':
            self.motion_sensitivity = 20
            self.min_motion_area = 200
        elif sensitivity_level == 'ultra':
            self.motion_sensitivity = 10
            self.min_motion_area = 100
        else:
            print(f"Invalid sensitivity level: {sensitivity_level}")
    
    def capture_motion_photo(self, frame, motion_area):
        """Capture photo when motion is detected and save to database"""
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"motion_{timestamp}.jpg"
            filepath = os.path.join(self.captures_dir, filename)
            
            # Save the frame as image
            cv2.imwrite(filepath, frame)
            
            # Calculate confidence based on motion area
            confidence = min(100, max(0, (motion_area / 1000) * 100))
            
            # Prepare motion data
            motion_data = {
                "motion_area": str(motion_area),
                "timestamp": timestamp,
                "sensitivity": self.motion_sensitivity,
                "min_area": self.min_motion_area
            }
            
            # Save to database
            result = save_motion_detection(
                motion_data=str(motion_data),
                confidence=str(confidence),
                captured_photo=filepath,
                device_serial="SNABC123",
                device_model="RPI3"
            )
            
            if result:
                print(f"Motion detected and saved: {filename} (Area: {motion_area}, Confidence: {confidence:.1f}%)")
            else:
                print(f"Failed to save motion detection to database: {filename}")
                
        except Exception as e:
            print(f"Error capturing motion photo: {e}")
    
    def get_frame(self):
        """Get current frame for streaming"""
        with self.frame_lock:
            if self.current_frame is not None:
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', self.current_frame)
                return buffer.tobytes()
        return None

# Initialize detector
detector = FaceMotionDetector()
# Set initial sensitivity to high for maximum motion detection
detector.adjust_sensitivity('high')

# Flask app
app = Flask(__name__)


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate():
        while True:
            frame = detector.get_frame()
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/motion_detection', methods=['POST'])
def motion_detection_endpoint():
    """Manual motion detection endpoint"""
    try:
        # Get current frame
        frame = detector.get_frame()
        if frame is None:
            return jsonify({"error": "No frame available"}), 400
        
        # Decode frame for processing
        nparr = np.frombuffer(frame, np.uint8)
        frame_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Simulate motion detection (you can modify this logic)
        motion_area = 1500  # Simulated motion area
        confidence = 85.5   # Simulated confidence
        
        # Capture photo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"manual_motion_{timestamp}.jpg"
        filepath = os.path.join(detector.captures_dir, filename)
        cv2.imwrite(filepath, frame_cv)
        
        # Prepare motion data
        motion_data = {
            "motion_area": str(motion_area),
            "timestamp": timestamp,
            "sensitivity": detector.motion_sensitivity,
            "min_area": detector.min_motion_area,
            "manual_trigger": True
        }
        
        # Save to database
        result = save_motion_detection(
            motion_data=str(motion_data),
            confidence=str(confidence),
            captured_photo=filepath,
            device_serial="SNABC123",
            device_model="RPI3"
        )
        
        if result:
            return jsonify({
                "success": True,
                "message": "Motion detection saved successfully",
                "filename": filename,
                "motion_area": motion_area,
                "confidence": confidence,
                "timestamp": timestamp
            })
        else:
            return jsonify({"error": "Failed to save to database"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/motion_status')
def motion_status():
    """Get current motion detection status"""
    return jsonify({
        "motion_detected": detector.motion_detected,
        "sensitivity": detector.motion_sensitivity,
        "min_motion_area": detector.min_motion_area,
        "last_motion_time": detector.last_motion_time,
        "motion_cooldown": detector.motion_cooldown
    })


if __name__ == '__main__':
    print("Starting Face Detection & Motion Sensor Application...")
    print("Features:")
    print("- Face detection with green bounding boxes")
    print("- High-sensitivity motion detection with red 'Motion' text in upper right")
    print("- Automatic photo capture and database storage on motion detection")
    print("- Flask web server for camera streaming")
    print("- Adjustable motion sensitivity levels")
    print("\nAccess URLs:")
    print("- Video stream: http://[PI_IP]:5000/video_feed")
    print("- Motion detection endpoint: POST http://[PI_IP]:5000/motion_detection")
    print("- Motion status: GET http://[PI_IP]:5000/motion_status")
    print("- Device Serial: SNABC123, Model: RPI3")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
        detector.camera.stop()
        detector.camera.close()
