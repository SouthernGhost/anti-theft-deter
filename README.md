# Anti-Theft Deterrent System

A comprehensive computer vision system for retail theft prevention using YOLO object detection. The system uses a pre-trained YOLOv8n model trained on Open Images V7 dataset for real-time monitoring of retail environments.

## Overview

This repository contains:
1. **Monitoring System**: Real-time theft detection and alert system using YOLO11n
2. **Pre-configured Classes**: Uses Open Images V7 classes for merchandise and person detection
3. **Audio Alert System**: Configurable audio announcements for theft deterrence
4. **Training Tools**: Optional custom training pipeline for specialized datasets

## Project Structure

```
anti-theft-deter/
├── audio/                 # Audio alert files (speech.wav)
├── videos/                # Test video files
├── config.py              # System configuration file
├── main.py                # Real-time monitoring system
├── train_yolo.py          # Optional custom training script
├── yolo11n.pt             # YOLO11n model trained on COCO8
└── README.md              # This documentation
```


### Configured Class Categories

1. **Person Classes**: Class ID 0 (expandable list in `PERSON_IDS_OIV7` in `config.py`)
2. **Merchandise Classes**: 200+ retail item class IDs from Open Images V7
   - Food items, beverages, containers, bags, electronics
   - Clothing, accessories, household items, tools
   - Personal care products, toys, sports equipment

### Class Configuration

The system uses pre-defined class ID lists in `config.py`:
- `PERSON_IDS_COCO = [0]` - Person detection classes
- `MERCH_IDS_COCO = [24, 25, 26, ...]` - 200+ merchandise class IDs from Open Images V7

## Installation

### Prerequisites

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLO

### Install Dependencies

```bash
pip install ultralytics opencv-python numpy
```

### Additional Windows Dependencies

For audio alerts on Windows:
```bash
# winsound is included with Python on Windows
# No additional installation required
```

### Model Download

The system uses YOLO11n model:
```bash
# The model will be automatically downloaded when first used
# Or manually download: yolo11n.pt
```

## Usage

### 1. Real-Time Monitoring System (Primary Use)

#### Run the Monitoring System

```bash
python main.py
```

#### System Features

- **Multi-threaded Processing**: Efficient real-time video processing
- **Object Detection**: Detects people and retail merchandise using Open Images classes
- **Alert System**: Audio and visual alerts for suspicious activity
- **Bathroom Zone Monitoring**: Configurable zone monitoring for high-risk areas

#### Monitoring Capabilities

- Person detection using Open Images V7 classes
- Merchandise detection (200+ retail item categories from Open Images V7)
- Configurable detection zones (bathroom monitoring)
- Real-time audio announcements for detected events
- Visual bounding boxes with class labels

#### Configuration

All settings are now centralized in the CONFIG dictionary in the `config.py` file:

```python
CONFIG = {
    # Model Configuration
    "model_path": "yolo11n.pt",  # Path to YOLO model

    # Video Source Configuration
    "stream_mode": False,  # Set to True to use IP camera stream
    "video_source": "videos/vid13.mp4",  # Video file path, webcam (0), or IP camera URL
    "ip_camera_url": "http://192.168.10.9:8080/video",  # IP camera URL (used when stream_mode=True)

    # Detection Zone Configuration
    "bathroom_zone": {
        'x1': 0.01, 'y1': 0.7,   # Top-left corner (relative coordinates 0-1)
        'x2': 0.99, 'y2': 0.99   # Bottom-right corner (relative coordinates 0-1)
    },

    # Display Configuration
    "show_stats": False,  # Show statistics overlay
    "stats_scale_factor": 0.2,  # Scale factor for UI elements (0.1-1.0)

    # Annotation Toggles - Simple and clean
    "annotations": {
        "bathroom_zone": True,     # Show/hide bathroom zone rectangle and label
        "persons": True,           # Show/hide person bounding boxes and labels
        "items": True,             # Show/hide item/merchandise bounding boxes and labels
    },

    # Detection Classes
    "merchandise_classes": MERCH_IDS_COCO,

    "person_classes": PERSON_IDS_COCO,  # Person class IDs - add more if needed

    # Stream Configuration
    "max_reconnect_attempts": 10,
    "reconnect_delay": 5,  # seconds

    # Detection Configuration
    "confidence_threshold": 0.3,
    "max_detections": 50
}
```

### Annotation Toggles

The system includes simple and clean annotation controls:

#### Annotation Controls

The `annotations` dictionary provides independent control over display elements:

##### 1. `bathroom_zone` Toggle
- **True**: Shows yellow bathroom zone rectangle and label
- **False**: Hides zone visuals (detection logic still uses zone coordinates)
- **Use Case**: Hide for clean public displays while maintaining functionality

##### 2. `persons` Toggle
- **True**: Shows green bounding boxes around detected people
- **False**: Hides person boxes (people still detected for alarms)
- **Use Case**: Hide for privacy or focus on items only

##### 3. `items` Toggle
- **True**: Shows red bounding boxes around detected items/merchandise
- **False**: Hides item boxes (items still detected for alarms)
- **Use Case**: Hide for clean view while maintaining security monitoring

### Configuration Examples

#### Display Scenarios

```python
# Full Monitoring (Default)
"annotations": {
    "bathroom_zone": True,
    "persons": True,
    "items": True
}

# Clean Public Display
"annotations": {
    "bathroom_zone": False,
    "persons": False,
    "items": False
}

# Zone Compliance Only
"annotations": {
    "bathroom_zone": True,
    "persons": False,
    "items": True
}

# People Tracking Only
"annotations": {
    "bathroom_zone": True,
    "persons": True,
    "items": False
}

# Privacy Mode
"annotations": {
    "bathroom_zone": False,
    "persons": False,
    "items": True
}

# Zone Boundary Only
"annotations": {
    "bathroom_zone": True,
    "persons": False,
    "items": False
}
```

## File Outputs

### Monitoring Outputs

- Real-time video display with detection overlays and bathroom zone
- Console logs of detected events with class IDs and confidence scores
- Audio alerts (speech1.wav) for suspicious activities


## Performance Optimization

### Real-time Monitoring Performance

- **GPU Acceleration**: Use CUDA-compatible GPU for faster inference
- **Video Resolution**: Reduce input video resolution for better performance
- **Detection Frequency**: Adjust monitoring thread sleep time (currently 1.0 seconds)
- **Confidence Threshold**: Increase confidence threshold to reduce false positives
- **Max Detections**: Reduce `max_det` parameter to limit processing load

### Configuration Examples

```python
# In main.py - for better performance
results = model.predict(
    frame,
    classes=self.merchandise_classes + self.person_classes,
    conf=self.config["confidence_threshold"],  # Increase from 0.1 for fewer false positives
    verbose=False,
    max_det=self.config["max_detections"]  # Reduce from 600 for faster processing
)

# Adjust monitoring frequency
time.sleep(2.0)  # Increase from 1.0 for less frequent checks
```

## Troubleshooting 

### Detection Issues

1. **No Person Detections**
   - Check if people are visible in the video frame
   - Check if person class IDs are correct for YOLO11n
   - Lower confidence threshold in detection
 
2. **No Merchandise Detections**
   - Verify merchandise class IDs are correct for Open Images V7
   - Lower confidence threshold in detection
   - Check if retail items are visible in the video frame
   - Review debug output for detected classes

3. **Model Loading Issues**
   - Ensure `yolo11n.pt` model file exists
   - Check internet connection for automatic model download
   - Verify model path in `config.py` configuration

### Performance Issues

1. **Slow Detection**
   - Use CPU device explicitly: `device='cpu'`
   - Reduce video resolution
   - Close other applications

2. **Memory Issues**
   - Reduce `max_det` parameter in prediction
   - Use smaller input image size
   - Monitor system memory usage

3. **Audio Issues**
   - Check Windows audio settings
   - Verify `speech1.wav` file exists
   - Test with `winsound.Beep()` fallback

## Advanced Usage

### Custom Training (Optional)

For specialized use cases, you can train custom models using the training script:

```bash
python train_yolo.py
```

This is optional since the system works well with the pre-trained YOLO11n model.

### Production Deployment

1. Download and verify `yolo11n.pt` model
2. Configure class IDs for your specific retail environment
3. Adjust detection zones and thresholds
4. Set up automated alert systems

## License and Credits

This system uses:
- Ultralytics YOLOv8 for object detection
- OpenCV for video processing
- Windows winsound for audio alerts

## Quick Start Guide

### For Running the Monitoring System

1. **Prepare Environment**:
   ```bash
   pip install ultralytics opencv-python numpy
   ```

2. **Download Model**:
   The YOLO11n model will be automatically downloaded on first use

3. **Configure Detection**:
   Edit `main.py` to customize:
   - Video source (file or webcam)
   - Detection classes (person_ids, merch_ids)
   - Bathroom zone coordinates

4. **Run Monitoring**:
   ```bash
   python main.py
   ```

5. **Monitor Output**:
   - Watch the video feed
   - Listen for audio alerts
   - Verify bounding boxes and labels

### Visual Output

- Real-time video with bounding boxes
- Yellow bathroom zone rectangle
- Green boxes for people (Class 381)
- Red boxes for merchandise (Various class IDs)
- Class labels with confidence scores