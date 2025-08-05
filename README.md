# Anti-Theft Deterrent System

A comprehensive computer vision system for retail theft prevention using YOLO object detection. The system combines custom dataset creation, model training, and real-time monitoring capabilities for retail environments.

## Overview

This repository contains:
1. **Dataset Creation Tools**: Merge multiple retail datasets into a unified format
2. **Training Pipeline**: Train custom YOLO models with checkpoint resuming
3. **Monitoring System**: Real-time theft detection and alert system
4. **Pre-trained Models**: Ready-to-use models for retail object detection

## Project Structure

```
anti-theft-deter/
├── main.py                 # Real-time monitoring system
├── train_yolo.py          # YOLO training script with checkpointing
├── create_custom_dataset.py # Dataset merger (if available)
├── datasets/              # Training datasets
│   ├── custom_data/       # Merged dataset (219 classes)
│   ├── shopcart/          # Shopping cart detection dataset
│   ├── grocery2/          # Vegetables and produce dataset
│   └── grozi/             # Retail products dataset (120 classes)
├── videos/                # Test video files
├── audio/                 # Audio alert files
└── yolo11n.pt            # Pre-trained YOLO model
```

## Dataset Information

### Merged Custom Dataset

The system uses a comprehensive dataset with 219 classes:

| Dataset Component | Classes | Images | Description |
|------------------|---------|---------|-------------|
| **COCO** | 80 | 0 | Preserved pretrained classes |
| **Shopcart** | 1 | 215 | Shopping cart detection |
| **Grocery2** | 18 | 1,731 | Vegetables and produce |
| **Grozi** | 120 | 11,194 | Retail products |
| **TOTAL** | **219** | **13,140** | Complete merged dataset |

### Class Categories

1. **COCO Classes (0-79)**: Standard objects (person, bicycle, car, etc.)
2. **Shopcart Classes (80)**: Shopping cart detection
3. **Grocery2 Classes (81-98)**: Fresh produce (Asparagus, Brinjal, Cabbage, etc.)
4. **Grozi Classes (99-218)**: Retail products (cleaning supplies, snacks, beverages, etc.)

## Installation

### Prerequisites

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLO

### Install Dependencies

```bash
pip install ultralytics opencv-python PyYAML Pillow requests numpy
```

### Additional Windows Dependencies

For audio alerts on Windows:
```bash
# winsound is included with Python on Windows
# No additional installation required
```

## Usage

### 1. Training Custom Models

#### Configuration-Based Training

The training script uses a simple configuration approach. Edit the CONFIG section in `train_yolo.py`:

```python
CONFIG = {
    # Dataset path - Path to your custom dataset YAML file
    "data_yaml_path": "datasets/custom_data/data.yaml",
    
    # Training parameters
    "epochs": 100,              # Number of training epochs
    "imgsz": 640,               # Input image size (640, 1280, etc.)
    "batch_size": 16,           # Batch size (reduce if memory errors)
    "device": "cpu",            # Device: "auto", "cpu", "0" (GPU 0), etc.
    
    # Checkpoint settings (for resuming training)
    "checkpoint_dir": "checkpoints",  # Directory for checkpoints (None to disable)
    
    # Model settings
    "model_name": "yolo11n.pt", # YOLO model variant
}
```

#### Run Training

```bash
python train_yolo.py
```

#### Training Features

- **Automatic Checkpoint Resuming**: Training automatically resumes from the last checkpoint if available
- **Custom Model Saving**: Best model is automatically saved as `yolo_custom.pt` in the root directory
- **Progress Tracking**: Clear feedback on training progress and checkpoint status
- **Flexible Configuration**: Easy parameter adjustment through CONFIG dictionary

#### Training Workflow

1. **First Run**: Starts from epoch 0 with pretrained weights
   ```
   Checkpoint directory ready: checkpoints
   No existing checkpoint found, will start from epoch 0
   Starting training from epoch 0 with yolo11n.pt pretrained weights
   ```

2. **Subsequent Runs**: Automatically resumes from checkpoint
   ```
   Found existing checkpoint: checkpoints/last.pt
   Resuming training from checkpoint: checkpoints/last.pt
   ```

3. **Completion**: Saves custom model for easy access
   ```
   Best model saved as: yolo_custom.pt
   ```

### 2. Real-Time Monitoring System

#### Run the Monitoring System

```bash
python main.py
```

#### System Features

- **Multi-threaded Processing**: Efficient real-time video processing
- **Fisheye Camera Support**: Handles wide-angle surveillance cameras
- **Object Detection**: Detects people and retail merchandise
- **Alert System**: Audio and visual alerts for suspicious activity
- **Bathroom Monitoring**: Specialized monitoring for high-risk areas
- **Abandoned Item Detection**: Tracks items left behind by customers

#### Monitoring Capabilities

- Person detection and tracking
- Merchandise detection (219 product categories)
- Shopping cart monitoring
- Suspicious behavior alerts
- Real-time video display with bounding boxes
- Audio announcements for detected events

### 3. Model Inference

#### Using the Custom Trained Model

```bash
# Run inference on images
yolo predict model=yolo_custom.pt source=path/to/images

# Run inference on video
yolo predict model=yolo_custom.pt source=path/to/video.mp4

# Run inference on webcam
yolo predict model=yolo_custom.pt source=0
```

#### Using Pre-trained Models

```bash
# Use the base YOLO11n model
yolo predict model=yolo11n.pt source=path/to/images
```

## Configuration Options

### Training Configuration

Modify `train_yolo.py` CONFIG section:

```python
# Basic training settings
"epochs": 100,              # Training duration
"batch_size": 16,           # Memory usage control
"imgsz": 640,               # Input resolution

# Hardware settings
"device": "auto",           # "auto", "cpu", "0", "1", etc.

# Model selection
"model_name": "yolo11n.pt", # yolo11n.pt, yolo11s.pt, yolo11m.pt, etc.

# Checkpoint management
"checkpoint_dir": "checkpoints",  # Enable automatic resuming
```

### Monitoring Configuration

Edit `main.py` for monitoring settings:

- Camera input sources
- Detection thresholds
- Alert sensitivity
- Audio settings
- Display options

## File Outputs

### Training Outputs

- `yolo_custom.pt`: Custom trained model (root directory)
- `runs/detect/train/weights/best.pt`: Best model weights
- `runs/detect/train/weights/last.pt`: Latest checkpoint
- `checkpoints/last.pt`: Resumable checkpoint

### Monitoring Outputs

- Real-time video display with detection overlays
- Console logs of detected events
- Audio alerts for suspicious activities

## Performance Optimization

### Training Performance

- Use GPU for faster training: `"device": "0"`
- Adjust batch size based on available memory
- Use larger models for better accuracy: `yolo11s.pt`, `yolo11m.pt`
- Enable mixed precision training (automatic)

### Inference Performance

- Use smaller models for real-time: `yolo11n.pt`
- Reduce input resolution for speed: `imgsz=320`
- Use GPU acceleration when available
- Optimize video processing threads

## Troubleshooting

### Common Training Issues

1. **Memory Errors**: Reduce `batch_size` in CONFIG
2. **CUDA Errors**: Set `"device": "cpu"` for CPU training
3. **Dataset Not Found**: Check `data_yaml_path` in CONFIG
4. **Checkpoint Errors**: Delete `checkpoints/` folder to start fresh

### Common Monitoring Issues

1. **Camera Not Found**: Check camera index in `main.py`
2. **Model Loading Errors**: Ensure `yolo_custom.pt` exists
3. **Performance Issues**: Reduce video resolution or detection frequency
4. **Audio Errors**: Check Windows audio system and winsound module

## Advanced Usage

### Custom Dataset Creation

If you have the dataset creation script:

```bash
python create_custom_dataset.py --output custom_data --split 0.8
```

### Multi-GPU Training

```python
CONFIG = {
    "device": "0,1",  # Use multiple GPUs
    "batch_size": 32, # Increase batch size for multi-GPU
}
```

### Production Deployment

1. Train model with sufficient epochs (200-500)
2. Validate on test dataset
3. Deploy `yolo_custom.pt` to production system
4. Configure monitoring thresholds for environment
5. Set up automated alert systems

## License and Credits

This system uses:
- Ultralytics YOLO for object detection
- OpenCV for video processing
- PyTorch for deep learning
- Multiple retail datasets for comprehensive training

## Quick Start Guide

### For Training a Custom Model

1. **Prepare Environment**:
   ```bash
   pip install ultralytics opencv-python PyYAML Pillow requests numpy
   ```

2. **Configure Training**:
   Edit `train_yolo.py` CONFIG section with your desired settings

3. **Start Training**:
   ```bash
   python train_yolo.py
   ```

4. **Monitor Progress**:
   Training will automatically resume if interrupted

5. **Use Trained Model**:
   Your custom model will be saved as `yolo_custom.pt`

### For Running the Monitoring System

1. **Ensure Model Exists**:
   Either train a custom model or use the pre-trained `yolo11n.pt`

2. **Configure Camera**:
   Edit camera settings in `main.py` if needed

3. **Run Monitoring**:
   ```bash
   python main.py
   ```

4. **Monitor Output**:
   Watch the video feed and listen for audio alerts

## Dataset File Structure

### Custom Data Format

```
datasets/custom_data/
├── data.yaml              # Dataset configuration
├── train/
│   ├── images/            # Training images (10,512 files)
│   │   ├── grozi_1_video1.png
│   │   ├── shopcart_image1.jpg
│   │   └── grocery2_image1.jpg
│   └── labels/            # Training labels (10,512 files)
│       ├── grozi_1_video1.txt
│       ├── shopcart_image1.txt
│       └── grocery2_image1.txt
└── val/
    ├── images/            # Validation images (2,628 files)
    └── labels/            # Validation labels (2,628 files)
```

### Label Format

YOLO format labels (one per line):
```
class_id x_center y_center width height
```

Example:
```
0 0.5 0.5 0.3 0.4    # person at center, 30% width, 40% height
99 0.2 0.3 0.1 0.15  # grozi product at 20% x, 30% y
```

## Model Performance

### Training Metrics

After training completion, you'll see metrics like:
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95**: Mean Average Precision across IoU thresholds 0.5-0.95
- **Box Loss**: Bounding box regression loss
- **Class Loss**: Classification loss
- **DFL Loss**: Distribution Focal Loss

### Inference Speed

Model performance varies by hardware:
- **CPU**: ~50-100ms per image (yolo11n.pt)
- **GPU (RTX 3080)**: ~5-10ms per image (yolo11n.pt)
- **GPU (RTX 4090)**: ~3-7ms per image (yolo11n.pt)

## System Requirements

### Minimum Requirements

- **CPU**: Intel i5 or AMD Ryzen 5 (4+ cores)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space for datasets
- **Python**: 3.8 or higher

### Recommended Requirements

- **CPU**: Intel i7 or AMD Ryzen 7 (8+ cores)
- **GPU**: NVIDIA RTX 3060 or better (6GB+ VRAM)
- **RAM**: 32GB for large batch training
- **Storage**: SSD with 50GB+ free space

### GPU Support

CUDA-compatible NVIDIA GPUs are recommended for:
- Faster training (10-50x speedup)
- Real-time inference
- Larger batch sizes
- Higher resolution processing

## Support

For issues and questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure dataset paths are correct
4. Check hardware compatibility (GPU/CPU)
5. Review console output for specific error messages
