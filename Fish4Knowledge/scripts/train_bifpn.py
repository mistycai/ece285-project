"""
BiFPN Experiment Training Script for Underwater Fish Detection

This script implements the BiFPN (Bi-directional Feature Pyramid Network) 
experiment, replacing YOLOv8's standard PANet neck with a more advanced
weighted feature fusion mechanism.

The key innovation: BiFPN uses learnable weights for each input feature map,
allowing the model to dynamically prioritize high-quality features over
degraded ones - particularly beneficial for underwater images with blur
and backscatter.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- THE FIX: Define the project root path explicitly ---
project_root = Path('/content/Fish4Knowledge')  # Adjust for your environment
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

# Import our custom BiFPN module
from models.bifpn import BiFPN, BiFPNBlock, BiFPNLayer, register_bifpn

# --- Register BiFPN modules with ultralytics ---
register_bifpn()

# Also register directly in tasks
tasks.BiFPN = BiFPN
tasks.BiFPNBlock = BiFPNBlock
tasks.BiFPNLayer = BiFPNLayer

def print_model_info(model):
    """Print detailed model information"""
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*60)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Check for BiFPN modules
    bifpn_count = 0
    for name, module in model.model.named_modules():
        if 'BiFPN' in str(type(module)):
            bifpn_count += 1
            print(f"Found BiFPN module: {name}")
    
    print(f"BiFPN modules detected: {bifpn_count}")
    print("="*60)

def main():
    """Main function to run the BiFPN detection experiment."""
    
    # --- Configuration ---
    CUSTOM_MODEL_YAML = project_root / 'models' / 'yolov8n-custom-bifpn.yaml'
    DATA_YAML = project_root / 'data.yaml'
    EXPERIMENT_NAME = 'yolov8n_bifpn'
    BASELINE_WEIGHTS = '/content/Fish4Knowledge/runs/detect/yolov8n_baseline_fish4knowledge_sdk/weights/best.pt'
    
    # Enhanced augmentation for underwater conditions
    strong_aug_args = {
        'degrees': 15.0, 'translate': 0.2, 'scale': 0.9, 'shear': 10.0,
        'perspective': 0.001, 'hsv_h': 0.025, 'hsv_s': 0.9, 'hsv_v': 0.6,
        'flipud': 0.1, 'mixup': 0.15, 'copy_paste': 0.1,
    }

    print("="*70)
    print("STARTING BiFPN UNDERWATER FISH DETECTION EXPERIMENT")
    print("="*70)
    print("🐠 Experiment: Weighted Feature Fusion for Underwater Images")
    print("🔬 Innovation: BiFPN replaces PANet for smarter feature fusion")
    print("🎯 Goal: Better handling of blur/backscatter in underwater scenes")
    print("="*70)

    try:
        # 1. Load the model from custom architecture
        print(f"\nLoading custom model from: {CUSTOM_MODEL_YAML}")
        model = YOLO(str(CUSTOM_MODEL_YAML))

        model.load('yolov8n.pt')
        
        # 3. Print model information
        print_model_info(model)
        
        # 4. Start training
        print("\nStarting BiFPN training...")
        
        results = model.train(
            task='detect',
            data=str(DATA_YAML),
            epochs=100,
            imgsz=640,
            batch=16,
            workers=8,
            name=EXPERIMENT_NAME,
            patience=15,  # Slightly higher patience for complex model
            save_period=10,  # Save every 10 epochs
            # val=True,
            # plots=True,
            **strong_aug_args
        )
        
        print("\n BiFPN model training completed successfully!")
        
        # 5. Evaluate the final model
        print("\nEVALUATING FINAL BiFPN MODEL ON TEST SET")
        print("-" * 50)
        
        # Validation metrics
        val_results = model.val(data=str(DATA_YAML), split='test', plots=True)
        
        # Print key metrics
        if hasattr(val_results, 'results_dict'):
            metrics = val_results.results_dict
            print(f"Final Test Metrics:")
            print(f"mAP50: {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
            print(f"mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
            print(f"Precision: {metrics.get('metrics/precision(B)', 'N/A'):.4f}")
            print(f"Recall: {metrics.get('metrics/recall(B)', 'N/A'):.4f}")

        
        print("\n" + "="*50)
        print("🏁 BiFPN EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("💡 The BiFPN model should show improved performance on:")
        print("   • Blurry underwater images") 
        print("   • Images with backscatter")
        print("   • Complex underwater lighting conditions")
        print("   • Small fish detection in cluttered scenes")
        
        return model, results
        
    except Exception as e:
        print(f"\n❌ Error during BiFPN experiment: {e}")
        print("🔧 Troubleshooting tips:")
        print("   1. Check if bifpn.py is in the models/ directory")
        print("   2. Verify yolov8n-bifpn.yaml syntax")
        print("   3. Ensure data.yaml points to correct dataset")
        print("   4. Check GPU memory availability")
        raise