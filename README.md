# Vehicle Counter with YOLOv8

A real-time vehicle detection and counting system built with Python, Flask, and YOLOv8. This application can detect and count different types of vehicles (cars, motorcycles, buses, trucks, and bicycles) passing through a designated counting zone.

## Features

- Real-time vehicle detection using YOLOv8
- Counts multiple vehicle types:
  - Cars
  - Motorcycles
  - Buses
  - Trucks
  - Bicycles
- Web-based interface with live video feed
- Region-based counting system to avoid duplicate counts
- Visual feedback with bounding boxes and count display
- WebSocket integration for real-time count updates

## Requirements

- Python 3.8 or higher
- OpenCV
- Flask
- Flask-SocketIO
- PyTorch
- Ultralytics YOLOv8
- NumPy
- Pillow

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd vehicle-counter
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install flask flask-socketio opencv-python torch ultralytics numpy pillow
```

## Usage

1. Place your video file in the project directory and update the `video_source` variable in `app.py` (default is 'tes.mkv')

2. Run the application:
```bash
python app.py
```

3. Open your web browser and navigate to:
```
http://localhost:5000
```

## How It Works

1. **Vehicle Detection**: The system uses YOLOv8, a state-of-the-art object detection model, to detect vehicles in each frame.

2. **Counting Zone**: Two horizontal lines create a counting region in the middle of the frame (40-60% of frame height).

3. **Vehicle Tracking**: Each detected vehicle is assigned a unique ID based on its position and class.

4. **Counting Logic**: 
   - Vehicles are counted only when they enter the counting zone
   - Each vehicle is counted exactly once using the tracking system
   - Counts are updated in real-time and displayed on the video feed

## Configuration

You can modify the following parameters in `app.py`:

- `CONFIDENCE_THRESHOLD`: Detection confidence threshold (default: 0.35)
- `VEHICLE_CLASSES`: Types of vehicles to detect and count
- `video_source`: Input video source (file path or camera index)
- `device`: Processing device ('cuda' for GPU, 'cpu' for CPU)

## Web Interface

The web interface shows:
- Live video feed with detection boxes
- Vehicle counts for each category
- Counting zone boundaries (blue lines)
- Detection confidence scores

## Performance Notes

- GPU acceleration is supported if available (requires CUDA-compatible GPU)
- CPU mode is available for systems without dedicated GPU
- Processing speed depends on:
  - Input video resolution
  - Hardware capabilities
  - Number of detected objects

## Troubleshooting

1. If the video feed doesn't appear:
   - Check if the video file path is correct
   - Ensure the video format is supported
   - Verify webcam index if using camera input

2. If detection is slow:
   - Consider using GPU acceleration
   - Reduce input video resolution
   - Adjust confidence threshold

3. If counts are inaccurate:
   - Adjust the confidence threshold
   - Check if counting zone is appropriately positioned
   - Verify video quality and frame rate


## Kelompok 6 Matkul Pengembangan Aplikasi Sains Data Telkom University Surabaya