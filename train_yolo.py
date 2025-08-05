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
    "checkpoint_dir": "checkpoints",  # Directory to save/load checkpoints (None to disable)
                                      # If checkpoint exists, training will resume automatically
                                      # If no checkpoint, training starts from epoch 0

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

    # Setup checkpoint directory and check for existing checkpoint
    checkpoint_path = None
    resume_training = False

    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "last.pt"

        # Check if checkpoint exists
        if checkpoint_path.exists():
            resume_training = True
            print(f"🔍 Found existing checkpoint: {checkpoint_path}")
        else:
            print(f"📁 Checkpoint directory ready: {checkpoint_dir}")
            print(f"🆕 No existing checkpoint found, will start from epoch 0")

    print(f"🚀 Starting {CONFIG['model_name']} training...")
    print(f"   - Dataset: {data_yaml_path}")
    print(f"   - Epochs: {epochs}")
    print(f"   - Image size: {imgsz}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Device: {device}")
    if checkpoint_dir:
        print(f"   - Checkpoint: {'Resume' if resume_training else 'Start new'}")

    try:
        if resume_training:
            print(f"🔄 Resuming training from checkpoint: {checkpoint_path}")
            model = YOLO(str(checkpoint_path))
            results = model.train(
                data=str(data_yaml_path),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                device=device,
                resume=True
            )
        else:
            if checkpoint_path and checkpoint_path.exists():
                print(f"🚀 Starting new training from {CONFIG['model_name']} pretrained weights")
            else:
                print(f"🚀 Starting training from epoch 0 with {CONFIG['model_name']} pretrained weights")

            model = YOLO(CONFIG['model_name'])
            results = model.train(
                data=str(data_yaml_path),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                device=device
            )

        # Save checkpoint after training completes
        if checkpoint_dir:
            try:
                # Find the most recent training run directory
                runs_dir = Path("runs/detect")
                if runs_dir.exists():
                    train_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("train")]
                    if train_dirs:
                        latest_train_dir = max(train_dirs, key=lambda x: x.stat().st_mtime)
                        yolo_last_pt = latest_train_dir / "weights" / "last.pt"

                        if yolo_last_pt.exists():
                            shutil.copy(yolo_last_pt, checkpoint_path)
                            print(f"✅ Saved checkpoint to: {checkpoint_path}")
                        else:
                            print(f"⚠️  Could not find last.pt at: {yolo_last_pt}")
                    else:
                        print("⚠️  No training directories found for checkpoint saving")
                else:
                    print("⚠️  runs/detect directory not found for checkpoint saving")
            except Exception as e:
                print(f"⚠️  Failed to save checkpoint: {e}")
        
        # Copy best model to root folder with custom name
        best_model_path = None
        try:
            # Find the most recent training run directory
            runs_dir = Path("runs/detect")
            if runs_dir.exists():
                # Get all train directories and find the most recent one
                train_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("train")]
                if train_dirs:
                    # Sort by modification time to get the most recent
                    latest_train_dir = max(train_dirs, key=lambda x: x.stat().st_mtime)
                    best_model_source = latest_train_dir / "weights" / "best.pt"

                    if best_model_source.exists():
                        best_model_path = Path("yolo_custom.pt")
                        shutil.copy(best_model_source, best_model_path)
                        print(f"✅ Best model saved as: {best_model_path.resolve()}")
                    else:
                        print(f"⚠️  Best model not found at: {best_model_source}")
                else:
                    print("⚠️  No training directories found in runs/detect/")
            else:
                print("⚠️  runs/detect directory not found")
        except Exception as e:
            print(f"⚠️  Failed to copy best model: {e}")

        print("✅ Training completed successfully!")
        print(f"   - Model weights saved in: runs/detect/train/weights/")
        print(f"   - Best model: runs/detect/train/weights/best.pt")
        print(f"   - Last model: runs/detect/train/weights/last.pt")
        if best_model_path and best_model_path.exists():
            print(f"   - Custom model: {best_model_path.resolve()}")

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
        print(f"   yolo predict model=yolo_custom.pt source=path/to/images")
        print(f"   # Or use the original path:")
        print(f"   yolo predict model=runs/detect/train/weights/best.pt source=path/to/images")

    else:
        print(f"\n❌ Training failed!")

if __name__ == "__main__":
    main()
