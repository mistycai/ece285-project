import sys
from pathlib import Path
import torch
import torch.nn as nn

# --- THE FIX: Define the project root path explicitly for Colab ---
project_root = Path('/content/Fish4Knowledge')
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# ---

from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
# Assuming your cbam.py is in /content/Fish4Knowledge/models/
from models.cbam import CBAM  

# --- Monkey-Patch to register your custom module ---
tasks.CBAM = CBAM


def main():
    """Main function to run the CBAM detection experiment."""
    
    # --- Configuration ---
    # We now use the project_root variable to build the path
    CUSTOM_MODEL_YAML = project_root / 'models' / 'yolov8n-custom-cbam.yaml'
    DATA_YAML = project_root / 'data.yaml'
    EXPERIMENT_NAME = 'yolov8n_cbam'
    # ---

    print("="*50)
    print("STARTING CUSTOM CBAM MODEL TRAINING")
    print(f"Model Architecture: {CUSTOM_MODEL_YAML}")
    print("Custom CBAM module registered at runtime.")
    print("="*50)

    # 1. Load the model from your custom architecture
    model = YOLO(str(CUSTOM_MODEL_YAML))

    # 2. Load pre-trained weights from the standard yolov8n.pt file
    model.load('yolov8n.pt') 

    # 3. Train the model
    print("\nStarting training for custom CBAM model...")
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
        # You can add strong_aug_args here if you wish
    )

    print("\nCustom CBAM model training complete.")

    # 4. Evaluate the final model
    print("\n--- FINAL CUSTOM CBAM MODEL PERFORMANCE ON TEST SET ---")
    model.val(data=str(DATA_YAML), split='test')


# --- This guard allows the script to be importable and runnable ---
if __name__ == '__main__':
    # This part is for running from a terminal as 'python your_script.py'
    # In Colab, you will call main() directly as shown in the next step.
    main()