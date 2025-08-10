#!/usr/bin/env python3
"""
Main Pipeline Script for Graphene Segmentation
==============================================

This script orchestrates the entire pipeline:
1. Create quadrant dataset from original images
2. Preprocess images (color normalization, CLAHE, resizing)
3. Create augmented dataset
4. Train 2DMOINet model
5. Test and evaluate results

Based on the 2020 Han et al. paper techniques.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"Running: {script_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        
        elapsed_time = time.time() - start_time
        print(f"SUCCESS: {description} completed successfully in {elapsed_time:.2f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Error in {description}:")
        print(f"Error code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False





def create_output_directories():
    """Create necessary output directories"""
    print("\nCreating output directories...")
    
    directories = [
        "output",
        "quadrant_dataset",
        "quadrant_dataset/images",
        "quadrant_dataset/masks",
        "quadrant_dataset/processed_images",
        "quadrant_dataset/processed_masks",
        "augmented_dataset",
        "augmented_dataset/augmented_images",
        "augmented_dataset/augmented_masks",
        "test_results",
        "single_image_results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"CREATED: {directory}")

def main():
    """Main pipeline execution"""
    print("Graphene Segmentation Pipeline")
    print("Based on 2020 Han et al. Deep Learning for 2D Materials")
    print("=" * 60)
    
    

    
    # Create output directories
    create_output_directories()
    
    # Pipeline steps
    pipeline_steps = [
        ("create_quadrant_dataset.py", "Creating quadrant dataset from original images"),
        ("enhanced_preprocess.py", "Preprocessing images (color normalization, CLAHE, resizing)"),
        ("enhanced_augmentation.py", "Creating augmented dataset with data augmentation"),
        ("train_2dmoinet.py", "Training 2DMOINet model"),
        ("test_2dmoinet.py", "Testing and evaluating model performance")
    ]
    
    # Execute pipeline steps
    for script_name, description in pipeline_steps:
        if not run_script(script_name, description):
            print(f"ERROR: Pipeline failed at: {description}")
            return False
    
    print("\nSUCCESS: PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Results available in:")
    print("  - Trained model: graphene_2dmoinet_final.pth")
    print("  - Test results: test_results/")
    print("  - Single image results: single_image_results/")
    print("  - Training history: output/training_history.png")
    print("  - Augmented dataset: augmented_dataset/")
    
    return True

def run_single_step(step_name):
    """Run a single step of the pipeline"""
    step_map = {
        "quadrant": ("create_quadrant_dataset.py", "Creating quadrant dataset"),
        "preprocess": ("enhanced_preprocess.py", "Preprocessing images"),
        "augment": ("enhanced_augmentation.py", "Creating augmented dataset"),
        "train": ("train_2dmoinet.py", "Training model"),
        "test": ("test_2dmoinet.py", "Testing model")
    }
    
    if step_name not in step_map:
        print(f"ERROR: Unknown step: {step_name}")
        print(f"Available steps: {list(step_map.keys())}")
        return False
    
    script_name, description = step_map[step_name]
    return run_script(script_name, description)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Graphene Segmentation Pipeline")
    parser.add_argument("--step", type=str, help="Run single step (quadrant/preprocess/augment/train/test)")
    
    args = parser.parse_args()
    
    if args.step:
        # Run single step
        run_single_step(args.step)
    else:
        # Run full pipeline
        main() 