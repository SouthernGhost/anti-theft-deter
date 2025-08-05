#!/usr/bin/env python3
"""
Simple YOLO Training Script

This script trains a YOLO11 model on a pre-created custom dataset.
Use create_custom_dataset.py first to create the merged dataset.

Features:
- Trains on custom merged dataset
- Uses pretrained YOLO11n weights (preserves COCO classes)
- Configurable training parameters
- Resume training from checkpoints
- Simple and efficient

Usage:
    1. Create your custom dataset: python create_custom_dataset.py
    2. Modify the CONFIG section below with your desired settings
    3. Run: python train_yolo.py
"""

import shutil
from pathlib import Path
from ultralytics import YOLO

# ============================================================================
# CONFIGURATION - Modify these settings as needed
# ============================================================================

CONFIG = {
    # Dataset path - Path to your custom dataset YAML file
    # Change this to point to your dataset (created by create_custom_dataset.py)
    "data_yaml_path": "datasets/custom_data/data.yaml",

    # Training parameters
    "epochs": 100,              # Number of training epochs
    "imgsz": 640,               # Input image size (640, 1280, etc.)
    "batch_size": 16,           # Batch size (reduce if you get memory errors)
    "device": "cpu",            # Device: "auto", "cpu", "0" (GPU 0), "1" (GPU 1), etc.

    # Checkpoint settings (for resuming training)
    "checkpoint_dir": None,     # Directory to save/load checkpoints (None to disable)

    # Model settings
    "model_name": "yolo11n.pt", # YOLO model: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
}

# ============================================================================

def train_model(data_yaml_path, epochs=100, imgsz=640, batch_size=16, checkpoint_dir=None, device='auto'):
    """Train YOLO model on custom dataset"""

    data_yaml_path = Path(data_yaml_path)
    if not data_yaml_path.exists():
        print(f"❌ Data YAML file not found: {data_yaml_path}")
        print("💡 Run 'python create_custom_dataset.py' first to create the dataset")
        return None, None

    # Setup checkpoint directory
    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        last_ckpt = checkpoint_dir / "last.pt"
    else:
        last_ckpt = None

    print(f"🚀 Starting {CONFIG['model_name']} training...")
    print(f"   - Dataset: {data_yaml_path}")
    print(f"   - Epochs: {epochs}")
    print(f"   - Image size: {imgsz}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Device: {device}")

    try:
        # Resume training if checkpoint exists
        if last_ckpt and last_ckpt.exists():
            print(f"🔄 Resuming training from {last_ckpt}")
            model = YOLO(str(last_ckpt))
            results = model.train(
                data=str(data_yaml_path),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                device=device,
                resume=True
            )
        else:
            print(f"🚀 Starting new training from {CONFIG['model_name']} pretrained weights")
            model = YOLO(CONFIG['model_name'])
            results = model.train(
                data=str(data_yaml_path),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                device=device
            )
            
            # Save checkpoint if directory specified
            if checkpoint_dir:
                yolo_run_dir = Path("runs/detect/train/weights/last.pt")
                if yolo_run_dir.exists():
                    shutil.copy(yolo_run_dir, last_ckpt)
                    print(f"✅ Saved checkpoint to {last_ckpt}")
        
        print("✅ Training completed successfully!")
        print(f"   - Model weights saved in: runs/detect/train/weights/")
        print(f"   - Best model: runs/detect/train/weights/best.pt")
        print(f"   - Last model: runs/detect/train/weights/last.pt")
        
        return model, results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None, None

def validate_dataset(data_yaml_path):
    """Validate that the dataset exists and is properly formatted"""
    data_yaml_path = Path(data_yaml_path)
    
    if not data_yaml_path.exists():
        return False, f"Data YAML file not found: {data_yaml_path}"
    
    try:
        import yaml
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Check required fields
        required_fields = ['train', 'val', 'nc', 'names']
        for field in required_fields:
            if field not in data_config:
                return False, f"Missing required field '{field}' in data.yaml"
        
        # Check directories exist
        train_dir = Path(data_config['train'])
        val_dir = Path(data_config['val'])
        
        if not train_dir.exists():
            return False, f"Training directory not found: {train_dir}"
        
        if not val_dir.exists():
            return False, f"Validation directory not found: {val_dir}"
        
        # Count images
        train_images = len(list(train_dir.glob("*.jpg"))) + len(list(train_dir.glob("*.png")))
        val_images = len(list(val_dir.glob("*.jpg"))) + len(list(val_dir.glob("*.png")))
        
        if train_images == 0:
            return False, f"No training images found in {train_dir}"
        
        if val_images == 0:
            return False, f"No validation images found in {val_dir}"
        
        print(f"✅ Dataset validation passed:")
        print(f"   - Classes: {data_config['nc']}")
        print(f"   - Training images: {train_images}")
        print(f"   - Validation images: {val_images}")
        
        return True, "Dataset is valid"
        
    except Exception as e:
        return False, f"Error validating dataset: {e}"

def main():
    """Main training function"""
    print("🎯 YOLO11 Training Script")
    print("=" * 50)
    print("📋 Configuration:")
    print(f"   - Dataset: {CONFIG['data_yaml_path']}")
    print(f"   - Epochs: {CONFIG['epochs']}")
    print(f"   - Image size: {CONFIG['imgsz']}")
    print(f"   - Batch size: {CONFIG['batch_size']}")
    print(f"   - Device: {CONFIG['device']}")
    print(f"   - Model: {CONFIG['model_name']}")
    if CONFIG['checkpoint_dir']:
        print(f"   - Checkpoint dir: {CONFIG['checkpoint_dir']}")
    print()

    # Validate dataset
    print("🔍 Validating dataset...")
    is_valid, message = validate_dataset(CONFIG['data_yaml_path'])

    if not is_valid:
        print(f"❌ Dataset validation failed: {message}")
        if "not found" in message:
            print("💡 Run 'python create_custom_dataset.py' first to create the dataset")
        return

    # Start training
    print(f"\n🚀 Starting training...")
    model, results = train_model(
        data_yaml_path=CONFIG['data_yaml_path'],
        epochs=CONFIG['epochs'],
        imgsz=CONFIG['imgsz'],
        batch_size=CONFIG['batch_size'],
        device=CONFIG['device'],
        checkpoint_dir=CONFIG['checkpoint_dir']
    )
    
    if model and results:
        print(f"\n🎉 Training completed successfully!")
        print(f"📊 Results summary:")
        print(f"   - Dataset: {CONFIG['data_yaml_path']}")
        print(f"   - Epochs completed: {CONFIG['epochs']}")
        print(f"   - Model saved: runs/detect/train/weights/best.pt")

        # Show some metrics if available
        try:
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
                if 'metrics/mAP50(B)' in metrics:
                    print(f"   - mAP@0.5: {metrics['metrics/mAP50(B)']:.3f}")
                if 'metrics/mAP50-95(B)' in metrics:
                    print(f"   - mAP@0.5:0.95: {metrics['metrics/mAP50-95(B)']:.3f}")
        except:
            pass

        print(f"\n📝 To run inference:")
        print(f"   yolo predict model=runs/detect/train/weights/best.pt source=path/to/images")

    else:
        print(f"\n❌ Training failed!")

if __name__ == "__main__":
    main()
