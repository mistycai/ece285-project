# File: scripts/train_eca.py
import sys
from pathlib import Path

# --- Setup Paths ---
project_root = Path('/content/Fish4Knowledge')
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
# Import your new ECA class
from models.eca import ECA  

# --- Monkey-Patch to register your custom ECA module ---
tasks.ECA = ECA

def main():
    """Main function to run the ECA detection experiment."""
    
    # --- Configuration ---
    CUSTOM_MODEL_YAML = project_root / 'models' / 'yolov8n-custom-eca.yaml'
    DATA_YAML = project_root / 'data.yaml'
    EXPERIMENT_NAME = 'yolov8n_eca'
    # ---

    print("="*50)
    print("STARTING CUSTOM ECA MODEL TRAINING")
    print(f"Model Architecture: {CUSTOM_MODEL_YAML}")
    print("Custom ECA module registered at runtime.")
    print("="*50)

    # Load the model from your custom architecture
    model = YOLO(str(CUSTOM_MODEL_YAML))

    # Load pre-trained weights from the standard yolov8n.pt file
    model.load('yolov8n.pt') 
    strong_aug_args = {
        'degrees': 15.0, 'translate': 0.2, 'scale': 0.9, 'shear': 10.0,
        'perspective': 0.001, 'hsv_h': 0.025, 'hsv_s': 0.9, 'hsv_v': 0.6,
        'flipud': 0.1, 'mixup': 0.15, 'copy_paste': 0.1,
    }
    # Train the model
    model.train(
    task='detect',
    data=str(DATA_YAML),
    epochs=100,
    imgsz=640,
    batch=16,
    workers=8,
    name=EXPERIMENT_NAME,
    patience=10,
    **strong_aug_args # Apply the augmentations as well
)

    print("\nCustom ECA model training complete.")

    # Evaluate the final model
    print("\n--- FINAL CUSTOM ECA MODEL PERFORMANCE ON TEST SET ---")
    model.val(data=str(DATA_YAML), split='test')