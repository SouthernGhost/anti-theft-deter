# Bathroom Anti-Theft Monitoring System

A multi-threaded computer vision system designed to monitor bathroom entrances using a fisheye camera to detect customers carrying merchandise and deter theft.

## Features

### Core Functionality
- **Real-time Object Detection**: Uses YOLO11 to detect people and merchandise items
- **Multi-threaded Architecture**: Separate threads for video capture, object detection, and display
- **Bathroom Zone Monitoring**: Configurable zone detection for bathroom entrance areas
- **Merchandise Detection**: Identifies bags, backpacks, handbags, suitcases, and other items
- **Customer Scanning**: Automatically detects when customers approach the bathroom zone
- **Audio Announcements**: Text-to-speech warnings when merchandise is detected (optional)
- **Abandoned Item Tracking**: Monitors items left behind for extended periods
- **Real-time Statistics**: Tracks customers scanned, merchandise detected, and theft deterred

### Anti-Theft Features
- **Merchandise Alerts**: Warns customers that merchandise is not permitted in bathrooms
- **Abandoned Item Detection**: Identifies items left behind as potential theft attempts
- **Visual Monitoring**: Real-time display with annotated detections
- **Statistical Reporting**: Comprehensive tracking of security events

## Files

### Main Implementation
- **`demo2.py`**: Full multi-threaded bathroom monitoring system
- **`demo_simple.py`**: Simplified single-threaded version for testing and demonstration
- **`test_video.py`**: Basic video reading and YOLO detection test

### Supporting Files
- **`demo.py`**: Original detection system with tracking capabilities
- **`vid2.mp4`**: Sample video file for testing
- **`yolo11s.pt`**: YOLO11 small model weights (downloaded automatically)

## Installation

### Requirements
```bash
pip install ultralytics opencv-python numpy pyttsx3
```

### Optional Dependencies
- **pyttsx3**: For text-to-speech announcements (system will work without it)
- **TensorRT**: For optimized inference (optional)

## Usage

### Quick Start
```bash
# Run the simple demonstration
python demo_simple.py

# Run the full multi-threaded system
python demo2.py
```

### Configuration

#### Video Source
```python
# Use webcam
video_source = 0

# Use video file
video_source = "vid2.mp4"

# Use IP camera
video_source = "rtsp://camera_ip/stream"
```

#### Bathroom Zone
```python
bathroom_zone = {
    'x1': 0.3, 'y1': 0.2,  # Top-left corner (relative coordinates 0-1)
    'x2': 0.7, 'y2': 0.8   # Bottom-right corner (relative coordinates 0-1)
}
```

#### YOLO Model
```python
# Different model sizes available
model_path = "yolo11n.pt"  # Nano (fastest)
model_path = "yolo11s.pt"  # Small (balanced)
model_path = "yolo11m.pt"  # Medium (more accurate)
model_path = "yolo11l.pt"  # Large (most accurate)
```

## System Architecture

### Multi-threaded Design (demo2.py)
1. **Video Capture Thread**: Continuously captures frames from camera/video
2. **Detection Thread**: Runs YOLO inference on captured frames
3. **Display Thread**: Shows annotated frames with detections
4. **Monitoring Thread**: Processes detections for security logic

### Detection Classes
- **Person (Class 0)**: Customers approaching bathroom
- **Merchandise Classes**:
  - Class 24: Backpack
  - Class 25: Umbrella
  - Class 26: Handbag
  - Class 27: Tie
  - Class 28: Suitcase
  - Class 29: Frisbee

## Controls

### Keyboard Commands
- **'q'**: Quit the application
- **'s'**: Show current statistics
- **ESC**: Close OpenCV windows

### Visual Indicators
- **Green Box**: Person detected
- **Yellow Box**: Customer in bathroom zone
- **Red Box**: Merchandise detected
- **Magenta Box**: Merchandise in bathroom zone (ALERT)
- **Yellow Rectangle**: Bathroom monitoring zone

## Statistics Tracking

The system tracks the following metrics:
- **Customers Scanned**: Number of people detected in bathroom zone
- **Merchandise Detected**: Number of items found with customers
- **Announcements Made**: Number of audio warnings issued
- **Abandoned Items**: Items left behind for extended periods
- **Theft Deterred**: Successful prevention events

## Deployment Considerations

### Hardware Requirements
- **Camera**: Fisheye camera for wide-angle bathroom entrance monitoring
- **Processing**: GPU recommended for real-time YOLO inference
- **Storage**: Minimal storage needed (no video recording by default)

### Privacy and Legal
- Ensure compliance with local privacy laws
- Position camera to monitor entrance only, not bathroom interiors
- Consider adding privacy notices
- Implement data retention policies

### Performance Optimization
- Use TensorRT for faster inference on NVIDIA GPUs
- Adjust detection confidence thresholds based on environment
- Optimize bathroom zone coordinates for your specific layout
- Consider using smaller YOLO models for faster processing

## Customization

### Adding New Merchandise Classes
```python
# Add COCO class IDs for additional items
merchandise_classes = [24, 25, 26, 27, 28, 29, 67]  # Added 67 for cell phone
```

### Adjusting Detection Sensitivity
```python
# Lower confidence for more detections
results = model(frame, conf=0.3)

# Higher confidence for fewer false positives
results = model(frame, conf=0.7)
```

### Custom Announcement Messages
```python
def _make_announcement(self, item_count):
    if item_count == 1:
        message = "Please leave your item outside the bathroom."
    else:
        message = f"Please leave your {item_count} items outside the bathroom."
```

## Troubleshooting

### Common Issues
1. **Camera not accessible**: Check camera permissions and connections
2. **YOLO model download fails**: Ensure internet connection
3. **Poor detection accuracy**: Adjust lighting and camera positioning
4. **High CPU usage**: Consider using GPU acceleration or smaller model

### Performance Tips
- Use appropriate model size for your hardware
- Adjust frame processing rate if needed
- Optimize bathroom zone size and position
- Consider using video file for testing before live deployment

## License

This system is designed for retail security applications. Ensure compliance with local laws and regulations regarding surveillance and privacy.
