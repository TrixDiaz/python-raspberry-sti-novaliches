import cv2
import numpy as np
import threading
import time
from flask import Flask, Response, render_template_string
from picamera2 import Picamera2
import os

class FaceMotionDetector:
    def __init__(self):
        # Initialize camera
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
        
        # Motion detection variables
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.motion_detected = False
        self.motion_timer = 0
        self.motion_display_duration = 3  # seconds
        
        # Frame for streaming
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def process_frames(self):
        """Main processing loop for face detection and motion detection"""
        while True:
            try:
                # Capture frame from camera
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
        
        # Motion detection
        motion_mask = self.background_subtractor.apply(frame)
        motion_contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check for significant motion
        motion_detected = False
        for contour in motion_contours:
            if cv2.contourArea(contour) > 500:  # Threshold for motion detection
                motion_detected = True
                break
        
        # Update motion status
        if motion_detected:
            self.motion_detected = True
            self.motion_timer = time.time()
        elif time.time() - self.motion_timer > self.motion_display_duration:
            self.motion_detected = False
        
        # Display motion text in upper right corner
        if self.motion_detected:
            cv2.putText(frame, 'Motion', (frame.shape[1] - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame
    
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

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    """Main page with video stream"""
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Face Detection & Motion Sensor</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f0f0f0;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            h1 {
                text-align: center;
                color: #333;
                margin-bottom: 30px;
            }
            .video-container {
                text-align: center;
                margin-bottom: 20px;
            }
            img {
                max-width: 100%;
                height: auto;
                border: 2px solid #ddd;
                border-radius: 8px;
            }
            .info {
                background-color: #e8f4fd;
                padding: 15px;
                border-radius: 5px;
                margin-top: 20px;
            }
            .status {
                display: flex;
                justify-content: space-around;
                margin-top: 20px;
            }
            .status-item {
                text-align: center;
                padding: 10px;
                border-radius: 5px;
                background-color: #f8f9fa;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Face Detection & Motion Sensor</h1>
            <div class="video-container">
                <img src="{{ url_for('video_feed') }}" alt="Video Stream">
            </div>
            <div class="info">
                <h3>Features:</h3>
                <ul>
                    <li><strong>Face Detection:</strong> Green rectangles around detected faces</li>
                    <li><strong>Motion Detection:</strong> "Motion" text appears in upper right corner when motion is detected</li>
                    <li><strong>Real-time Streaming:</strong> Live camera feed via Flask web server</li>
                </ul>
            </div>
            <div class="status">
                <div class="status-item">
                    <h4>Face Detection</h4>
                    <p>Active - Green boxes show detected faces</p>
                </div>
                <div class="status-item">
                    <h4>Motion Detection</h4>
                    <p>Active - Red "Motion" text when movement detected</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    ''')

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

@app.route('/status')
def status():
    """API endpoint to get detection status"""
    return {
        'motion_detected': detector.motion_detected,
        'timestamp': time.time()
    }

if __name__ == '__main__':
    print("Starting Face Detection & Motion Sensor Application...")
    print("Features:")
    print("- Face detection with green bounding boxes")
    print("- Motion detection with red 'Motion' text in upper right")
    print("- Flask web server for camera streaming")
    print("\nAccess the application at: http://localhost:5000")
    print("Or use your Raspberry Pi's IP address: http://[PI_IP]:5000")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
        detector.camera.stop()
        detector.camera.close()
