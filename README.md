# Anti-Theft Deterrent System

A comprehensive computer vision system for retail theft prevention using YOLO object detection. The system uses a pre-trained YOLOv8n model trained on Open Images V7 dataset for real-time monitoring of retail environments.

## Overview

This repository contains:
1. **Monitoring System**: Real-time theft detection and alert system using YOLOv8n-OI7
2. **Pre-configured Classes**: Uses Open Images V7 classes for merchandise and person detection
3. **Audio Alert System**: Configurable audio announcements for theft deterrence
4. **Training Tools**: Optional custom training pipeline for specialized datasets

## Project Structure

```
anti-theft-deter/
├── main.py                 # Real-time monitoring system
├── test_detection.py       # Detection testing and debugging script
├── train_yolo.py          # Optional custom training script
├── videos/                # Test video files
├── audio/                 # Audio alert files (announcement.wav)
├── yolov8n-oi7.pt         # YOLOv8n model trained on Open Images V7
└── README.md              # This documentation
```

## Model Information

### YOLOv8n-OI7 Model

The system uses a pre-trained YOLOv8n model trained on Open Images V7 dataset:

| Model Component | Classes | Description |
|------------------|---------|-------------|
| **YOLOv8n-OI7** | 600+ | Pre-trained on Open Images V7 dataset |
| **Person Detection** | 1 | Class ID: 381 (configurable) |
| **Merchandise Detection** | 200+ | Pre-configured retail item class IDs |

### Configured Class Categories

1. **Person Classes**: Class ID 381 (expandable list in `person_ids`)
2. **Merchandise Classes**: 200+ retail item class IDs from Open Images V7
   - Food items, beverages, containers, bags, electronics
   - Clothing, accessories, household items, tools
   - Personal care products, toys, sports equipment

### Class Configuration

The system uses pre-defined class ID lists in `main.py`:
- `person_ids = [381]` - Person detection classes
- `merch_ids = [10, 16, 17, ...]` - 200+ merchandise class IDs from Open Images V7

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

The system uses YOLOv8n-OI7 model:
```bash
# The model will be automatically downloaded when first used
# Or manually download: yolov8n-oi7.pt
```

## Usage

### 1. Real-Time Monitoring System (Primary Use)

#### Run the Monitoring System

```bash
python main.py
```

#### System Features

- **Multi-threaded Processing**: Efficient real-time video processing
- **Open Images V7 Detection**: Uses pre-trained YOLOv8n-OI7 model with 600+ classes
- **Object Detection**: Detects people and retail merchandise using Open Images classes
- **Alert System**: Audio and visual alerts for suspicious activity
- **Bathroom Zone Monitoring**: Configurable zone monitoring for high-risk areas
- **Debug Mode**: Real-time class detection debugging and verification

#### Monitoring Capabilities

- Person detection using Open Images V7 classes
- Merchandise detection (200+ retail item categories from Open Images V7)
- Configurable detection zones (bathroom monitoring)
- Real-time audio announcements for detected events
- Visual bounding boxes with class labels
- Debug output for detection verification

#### Configuration

Edit `main.py` to customize:

```python
# Person class IDs (expandable list)
person_ids = [381]  # Add more person class IDs if needed

# Merchandise class IDs (200+ pre-configured)
merch_ids = [10, 16, 17, 21, 39, ...]  # Open Images V7 retail classes

# Bathroom zone coordinates (relative 0-1)
bathroom_zone = {
    'x1': 0.01, 'y1': 0.01,  # Top-left corner
    'x2': 0.99, 'y2': 0.6    # Bottom-right corner
}
```

### 2. Detection Testing and Debugging

#### Test Detection System

```bash
python test_detection.py
```

This script helps verify:
- Model loading and inference
- Class detection accuracy
- Person and merchandise detection
- Debug output for troubleshooting

### 3. Model Inference (Advanced)

#### Using the YOLOv8n-OI7 Model

```bash
# Run inference on images
yolo predict model=yolov8n-oi7.pt source=path/to/images

# Run inference on video
yolo predict model=yolov8n-oi7.pt source=path/to/video.mp4

# Run inference on webcam
yolo predict model=yolov8n-oi7.pt source=0
```

## Configuration Options

### Monitoring Configuration

Edit `main.py` for monitoring settings:

```python
# Model configuration
model_path = "yolov8n-oi7.pt"  # YOLOv8n Open Images V7 model

# Video source
video_source = "videos/vid5.mp4"  # Video file or 0 for webcam

# Detection classes
person_ids = [381]  # Person class IDs (expandable)
merch_ids = [10, 16, 17, ...]  # Merchandise class IDs

# Detection zone
bathroom_zone = {
    'x1': 0.01, 'y1': 0.01,  # Top-left (relative coordinates)
    'x2': 0.99, 'y2': 0.6    # Bottom-right (relative coordinates)
}

# Audio settings
show_stats = False  # Show/hide statistics overlay
stats_scale_factor = 0.2  # Scale factor for UI elements
```

### Class Configuration

To add more person or merchandise classes:

```python
# Add more person-related classes
person_ids = [381, 123, 456]  # Multiple person class IDs

# Merchandise classes are pre-configured from Open Images V7
# Edit merch_ids list to add/remove specific retail item classes
```

## File Outputs

### Monitoring Outputs

- Real-time video display with detection overlays and bathroom zone
- Console logs of detected events with class IDs and confidence scores
- Audio alerts (announcement.wav) for suspicious activities
- Debug output showing detected classes and detection statistics

### Debug Information

The system provides detailed debug output:
```
DEBUG: Person detected in zone - Class ID: 381, Conf: 0.85
DEBUG: Merchandise detected in zone - Class ID: 125, Conf: 0.72
DEBUG: All detected classes: [125, 381, 456, 789]
DEBUG: Looking for person classes: [381]
DEBUG: Looking for merchandise classes (first 10): [10, 16, 17, 21, 39, 57, 65, 67, 72, 76]...
ANNOUNCEMENT: Attention: Merchandise is not permitted in the bathroom...
```

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
    conf=0.3,  # Increase from 0.1 for fewer false positives
    verbose=False,
    max_det=300  # Reduce from 600 for faster processing
)

# Adjust monitoring frequency
time.sleep(2.0)  # Increase from 1.0 for less frequent checks
```

## Troubleshooting 

### Detection Issues

1. **No Person Detections**
   - Verify person class ID 381 is correct for YOLOv8n-OI7
   - Add more person class IDs to `person_ids` list
   - Check if people are visible in the video frame
   - Run `test_detection.py` to debug

2. **No Merchandise Detections**
   - Verify merchandise class IDs are correct for Open Images V7
   - Lower confidence threshold in detection
   - Check if retail items are visible in the video frame
   - Review debug output for detected classes

3. **Model Loading Issues**
   - Ensure `yolov8n-oi7.pt` model file exists
   - Check internet connection for automatic model download
   - Verify model path in `main.py` configuration

### Performance Issues

1. **Slow Detection**
   - Use CPU device explicitly: `device='cpu'`
   - Reduce video resolution
   - Close other applications

2. **Memory Issues**
   - Reduce `max_det` parameter in prediction
   - Use smaller input image size
   - Monitor system memory usage

### Audio Issues

1. **No Audio Alerts**
   - Check Windows audio settings
   - Verify `announcement.wav` file exists
   - Test with `winsound.Beep()` fallback

### Debug Mode

Enable debug output by running the system and checking console for:
- Detected class IDs
- Person/merchandise detection in zone
- Model loading status
- Audio alert triggers

## Advanced Usage

### Custom Training (Optional)

For specialized use cases, you can train custom models using the training script:

```bash
python train_yolo.py
```

This is optional since the system works well with the pre-trained YOLOv8n-OI7 model.

### Production Deployment

1. Download and verify `yolov8n-oi7.pt` model
2. Configure class IDs for your specific retail environment
3. Adjust detection zones and thresholds
4. Set up automated alert systems
5. Test detection accuracy with `test_detection.py`

## License and Credits

This system uses:
- Ultralytics YOLOv8 for object detection
- OpenCV for video processing
- Open Images V7 dataset for pre-trained classes
- Windows winsound for audio alerts

## Quick Start Guide

### For Running the Monitoring System

1. **Prepare Environment**:
   ```bash
   pip install ultralytics opencv-python numpy
   ```

2. **Download Model**:
   The YOLOv8n-OI7 model will be automatically downloaded on first use

3. **Configure Detection**:
   Edit `main.py` to customize:
   - Video source (file or webcam)
   - Detection classes (person_ids, merch_ids)
   - Bathroom zone coordinates

4. **Test Detection**:
   ```bash
   python test_detection.py
   ```

5. **Run Monitoring**:
   ```bash
   python main.py
   ```

6. **Monitor Output**:
   Watch the video feed and listen for audio alerts

## System Output Examples

### Console Output

```
Loading model: yolov8n-oi7.pt
Model loaded successfully: <class 'ultralytics.models.yolo.model.YOLO'>
DEBUG: Person detected in zone - Class ID: 381, Conf: 0.85
DEBUG: Merchandise detected in zone - Class ID: 125, Conf: 0.72
ANNOUNCEMENT: Attention: Merchandise is not permitted in the bathroom...
DEBUG: All detected classes: [125, 381, 456, 789]
```

### Visual Output

- Real-time video with bounding boxes
- Yellow bathroom zone rectangle
- Green boxes for people (Class 381)
- Red boxes for merchandise (Various class IDs)
- Class labels with confidence scores
## Model Performance

### YOLOv8n-OI7 Specifications

- **Model Size**: ~6MB (YOLOv8n architecture)
- **Classes**: 600+ from Open Images V7 dataset
- **Input Size**: 640x640 pixels
- **Framework**: PyTorch/Ultralytics

### Inference Speed

Model performance varies by hardware:
- **CPU**: ~50-100ms per image (YOLOv8n-OI7)
- **GPU (RTX 3080)**: ~5-10ms per image (YOLOv8n-OI7)
- **GPU (RTX 4090)**: ~3-7ms per image (YOLOv8n-OI7)

## System Requirements

### Minimum Requirements

- **CPU**: Intel i5 or AMD Ryzen 5 (4+ cores)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB free space for model and videos
- **Python**: 3.8 or higher
- **OS**: Windows (for winsound audio support)

### Recommended Requirements

- **CPU**: Intel i7 or AMD Ryzen 7 (8+ cores)
- **RAM**: 16GB for smooth real-time processing
- **Storage**: SSD with 10GB+ free space
- **Webcam**: USB camera for live monitoring

### GPU Support

CUDA-compatible NVIDIA GPUs are recommended for:
- Faster inference (5-10x speedup)
- Real-time processing of high-resolution video
- Multiple camera streams
- Reduced CPU usage

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Run `test_detection.py` to debug detection issues
3. Verify all dependencies are installed correctly
4. Check model file exists (`yolov8n-oi7.pt`)
5. Review console debug output for detection information
6. Ensure video source is accessible
7. Check hardware compatibility (GPU/CPU)
