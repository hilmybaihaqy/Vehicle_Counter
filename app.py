from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from PIL import Image
import io
import base64
import threading
import queue
import time

app = Flask(__name__)
socketio = SocketIO(app)

# Check if CUDA is available
device = 'cpu' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialize YOLO model
model = YOLO('yolov8n.pt')
model.to(device)  # Move model to GPU if available

# Print available classes
print("Available YOLO classes:")
print(model.names)

# Global variables
video_source = 'tes.mkv'  # Default webcam
frame_queue = queue.Queue(maxsize=10)

# Configuration
CONFIDENCE_THRESHOLD = 0.35  # Lower threshold to detect more objects
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    1: 'bicycle'
}

# Update vehicle counts to match YOLO classes
vehicle_counts = {
    'car': 0,        # class 2
    'motorcycle': 0, # class 3
    'bus': 0,        # class 5
    'truck': 0,      # class 7
    'bicycle': 0     # class 1
}

# Tracking variables
tracked_vehicles = {}  # Store tracked vehicles
counting_region = None  # Will be initialized with first frame
counted_vehicles = set()  # Store IDs of counted vehicles

def get_vehicle_id(bbox, cls):
    """Generate a unique ID for a vehicle based on its position and class"""
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return f"{cls}_{int(center_x)}_{int(center_y)}"

def process_frame(frame):
    try:
        global counting_region, tracked_vehicles
        
        if frame is None or frame.size == 0:
            print("Error: Received empty frame")
            return None
            
        # Initialize counting region if not set (middle 60% of frame height)
        if counting_region is None:
            height = frame.shape[0]
            counting_region = (int(height * 0.4), int(height * 0.6))
            
        # Draw counting region
        cv2.line(frame, (0, counting_region[0]), (frame.shape[1], counting_region[0]), (255, 0, 0), 2)
        cv2.line(frame, (0, counting_region[1]), (frame.shape[1], counting_region[1]), (255, 0, 0), 2)
            
        # Convert frame to RGB for YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run YOLO detection
        results = model(frame_rgb, device=device, conf=CONFIDENCE_THRESHOLD)
        
        # Current frame's vehicles
        current_vehicles = set()
        
        # Process results
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                cls_idx = int(box.cls[0])
                
                if cls_idx not in VEHICLE_CLASSES:
                    continue
                    
                cls = VEHICLE_CLASSES[cls_idx]
                conf = float(box.conf[0])
                
                if conf > CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = box.xyxy[0]
                    bbox = [int(x1), int(y1), int(x2), int(y2)]
                    
                    # Calculate center point
                    center_y = (bbox[1] + bbox[3]) / 2
                    
                    # Generate vehicle ID
                    vehicle_id = get_vehicle_id(bbox, cls)
                    current_vehicles.add(vehicle_id)
                    
                    # Check if vehicle is in counting region and not counted
                    if (counting_region[0] <= center_y <= counting_region[1] and 
                        vehicle_id not in counted_vehicles):
                        counted_vehicles.add(vehicle_id)
                        vehicle_counts[cls] += 1
                        print(f"Counted new {cls} with ID {vehicle_id}")
                    
                    # Draw bounding box
                    cv2.rectangle(frame, 
                                (bbox[0], bbox[1]), 
                                (bbox[2], bbox[3]), 
                                (0, 255, 0), 2)
                    
                    # Add label
                    label = f'{cls} {conf:.2f}'
                    cv2.putText(frame, 
                              label, 
                              (bbox[0], bbox[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.9,
                              (0, 255, 0),
                              2)
        
        # Clean up old vehicles
        tracked_vehicles = current_vehicles
        
        # Draw counts on frame
        y_pos = 30
        for vehicle_type, count in vehicle_counts.items():
            cv2.putText(frame,
                       f"{vehicle_type}: {count}",
                       (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       1,
                       (255, 255, 255),
                       2)
            y_pos += 30
        
        return frame
        
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def video_stream():
    global video_source
    cap = cv2.VideoCapture(video_source)
    retry_count = 0
    max_retries = 3
    
    print(f"Attempting to open video source: {video_source}")
    
    while retry_count < max_retries:
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}")
            retry_count += 1
            time.sleep(1)  # Wait before retrying
            cap = cv2.VideoCapture(video_source)
        else:
            print("Successfully opened video source!")
            break
    
    if not cap.isOpened():
        print("Failed to open video source after maximum retries")
        return

    frame_count = 0
    while True:
        try:
            success, frame = cap.read()
            frame_count += 1
            
            if frame_count % 30 == 0:  # Print status every 30 frames
                print(f"Frame {frame_count}: Read success = {success}, Frame shape = {frame.shape if success else 'None'}")
            
            if not success:
                print("Error: Could not read frame")
                # Try to reopen the stream
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(video_source)
                continue
            
            if frame is None or frame.size == 0:
                print("Error: Empty frame received")
                continue
                
            # Process frame with YOLO
            processed_frame = process_frame(frame)
            if processed_frame is None:
                print("Error: Frame processing failed")
                continue
            
            # Convert frame to base64 for sending to frontend
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = base64.b64encode(buffer).decode('utf-8')
            
            # Update counts via WebSocket
            socketio.emit('vehicle_counts', vehicle_counts)
            
            # Put frame in queue
            if not frame_queue.full():
                frame_queue.put(frame_bytes)
                
            # Small delay to prevent overwhelming the system
            time.sleep(0.01)
            
        except Exception as e:
            print(f"Error in video stream: {str(e)}")
            time.sleep(1)  # Wait before retrying
            continue
    
    cap.release()

def get_frame():
    while True:
        frame = frame_queue.get()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + base64.b64decode(frame) + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(get_frame(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start video processing in a separate thread
    video_thread = threading.Thread(target=video_stream)
    video_thread.daemon = True
    video_thread.start()
    
    # Run the Flask app with host='0.0.0.0' to make it accessible from other machines
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True) 