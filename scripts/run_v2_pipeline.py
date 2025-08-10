#!/usr/bin/env python3
"""
V2 Enhanced 2DMOINet Pipeline Runner
This script orchestrates the entire V2 training and testing pipeline
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_step(step_num, total_steps, description):
    """Print a formatted step description"""
    print(f"\n[Step {step_num}/{total_steps}] {description}")
    print("-" * 50)

def run_command(command, description, check=True):
    """Run a shell command and handle errors"""
    print(f"Running: {command}")
    print(f"Description: {description}")
    
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print("Output:", result.stdout)
        if result.stderr:
            print("Errors/Warnings:", result.stderr)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        if e.stdout:
            print("Output:", e.stdout)
        if e.stderr:
            print("Errors:", e.stderr)
        return False

def check_dependencies():
    """Check if required directories and files exist"""
    print_header("DEPENDENCY CHECK")
    
    required_dirs = [
        'data/raw/images',
        'data/raw/masks',
        'data/raw/annotations',
        'models/v2_enhanced',
        'scripts'
    ]
    
    required_files = [
        'models/v2_enhanced/dataset.py',
        'models/v2_enhanced/transforms.py',
        'models/v2_enhanced/losses.py',
        'models/v2_enhanced/model_architectures.py',
        'models/v2_enhanced/train_2dmoinet_v2.py',
        'models/v2_enhanced/test_2dmoinet_v2.py'
    ]
    
    missing_dirs = []
    missing_files = []
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_dirs or missing_files:
        print("Missing dependencies:")
        if missing_dirs:
            print("  Directories:", missing_dirs)
        if missing_files:
            print("  Files:", missing_files)
        return False
    
    print("All dependencies are satisfied!")
    return True

def prepare_data():
    """Prepare the dataset for V2 training"""
    print_header("DATA PREPARATION")
    
    # Step 1: Create quadrant dataset
    print_step(1, 3, "Creating quadrant dataset (4x4 = 16 images)")
    if not run_command(
        "python utils/create_quadrant_dataset.py",
        "Splitting original images into 4 quadrants"
    ):
        print("Failed to create quadrant dataset")
        return False
    
    # Step 2: Apply enhanced augmentation
    print_step(2, 3, "Applying enhanced augmentation (16x4 = 64 total images)")
    if not run_command(
        "python utils/enhanced_augmentation.py",
        "Applying advanced augmentation techniques"
    ):
        print("Failed to apply augmentation")
        return False
    
    # Step 3: Verify dataset
    print_step(3, 3, "Verifying final dataset")
    aug_images_dir = "data/augmented/augmented_dataset/augmented_images"
    aug_masks_dir = "data/augmented/augmented_dataset/augmented_masks"
    
    if os.path.exists(aug_images_dir) and os.path.exists(aug_masks_dir):
        num_images = len([f for f in os.listdir(aug_images_dir) if f.endswith(('.jpg', '.png'))])
        num_masks = len([f for f in os.listdir(aug_masks_dir) if f.endswith('.png')])
        print(f"Dataset prepared successfully!")
        print(f"  - Augmented images: {num_images}")
        print(f"  - Augmented masks: {num_masks}")
        return True
    else:
        print("Dataset preparation failed - augmented directories not found")
        return False

def train_model():
    """Train the V2 model"""
    print_header("MODEL TRAINING")
    
    print_step(1, 1, "Training V2 Enhanced 2DMOINet")
    
    # Create checkpoints directory
    os.makedirs("checkpoints/v2", exist_ok=True)
    
    # Run training
    if not run_command(
        "python models/v2_enhanced/train_2dmoinet_v2.py",
        "Training V2 model with enhanced techniques"
    ):
        print("Training failed")
        return False
    
    # Check if training completed successfully
    checkpoint_dir = "checkpoints/v2"
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if checkpoints:
            print(f"Training completed successfully!")
            print(f"Checkpoints saved: {checkpoints}")
            return True
    
    print("Training failed - no checkpoints found")
    return False

def test_model():
    """Test the trained V2 model"""
    print_header("MODEL TESTING")
    
    print_step(1, 1, "Evaluating V2 model performance")
    
    # Create results directory
    os.makedirs("results/v2", exist_ok=True)
    
    # Run testing
    if not run_command(
        "python models/v2_enhanced/test_2dmoinet_v2.py",
        "Evaluating V2 model on test dataset"
    ):
        print("Testing failed")
        return False
    
    # Check if testing completed successfully
    results_dir = "results/v2"
    if os.path.exists(results_dir):
        results_files = [f for f in os.listdir(results_dir) if f.endswith(('.png', '.json'))]
        if results_files:
            print(f"Testing completed successfully!")
            print(f"Results saved: {results_files}")
            return True
    
    print("Testing failed - no results found")
    return False

def generate_report():
    """Generate a summary report of the V2 pipeline"""
    print_header("PIPELINE SUMMARY REPORT")
    
    # Check final results
    results_dir = "results/v2"
    checkpoints_dir = "checkpoints/v2"
    
    print("V2 Pipeline Results:")
    print("-" * 30)
    
    if os.path.exists(checkpoints_dir):
        checkpoints = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pth')]
        print(f"Model checkpoints: {len(checkpoints)}")
        for ckpt in checkpoints:
            print(f"  - {ckpt}")
    
    if os.path.exists(results_dir):
        results_files = [f for f in os.listdir(results_dir) if f.endswith(('.png', '.json'))]
        print(f"Result files: {len(results_files)}")
        for result in results_files:
            print(f"  - {result}")
    
    # Dataset info
    if os.path.exists("aug_images"):
        num_images = len([f for f in os.listdir("aug_images") if f.endswith(('.jpg', '.png'))])
        print(f"Final dataset size: {num_images} images")
    
    print("\nV2 Enhancements Implemented:")
    print("-" * 30)
    enhancements = [
        "Advanced data augmentation (elastic deformation, grid distortion)",
        "Enhanced loss functions (Focal, Dice, Boundary, Combined)",
        "Improved model architecture with attention mechanisms",
        "Advanced learning rate scheduling (CosineAnnealingWarmRestarts)",
        "Mixed precision training",
        "Gradient clipping",
        "Early stopping",
        "Comprehensive metrics tracking"
    ]
    
    for i, enhancement in enumerate(enhancements, 1):
        print(f"  {i}. {enhancement}")
    
    print("\nNext Steps:")
    print("-" * 30)
    print("1. Review training curves and metrics")
    print("2. Analyze confusion matrix for class performance")
    print("3. Compare V2 results with V1 baseline")
    print("4. Consider additional improvements for V3")

def main():
    """Main pipeline execution"""
    print_header("V2 ENHANCED 2DMOINET PIPELINE")
    print("This pipeline implements advanced techniques from the research paper")
    print("to improve graphene layer segmentation performance.")
    
    start_time = time.time()
    
    # Check dependencies
    if not check_dependencies():
        print("Pipeline cannot continue due to missing dependencies")
        return
    
    # Execute pipeline steps
    steps = [
        ("Data Preparation", prepare_data),
        ("Model Training", train_model),
        ("Model Testing", test_model),
        ("Report Generation", generate_report)
    ]
    
    successful_steps = 0
    total_steps = len(steps)
    
    for step_name, step_func in steps:
        print_header(f"STARTING: {step_name.upper()}")
        
        if step_func():
            successful_steps += 1
            print(f"✅ {step_name} completed successfully")
        else:
            print(f"❌ {step_name} failed")
            print("Pipeline will continue with remaining steps...")
    
    # Final summary
    end_time = time.time()
    duration = end_time - start_time
    
    print_header("PIPELINE COMPLETION SUMMARY")
    print(f"Steps completed: {successful_steps}/{total_steps}")
    print(f"Total duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    
    if successful_steps == total_steps:
        print("🎉 All pipeline steps completed successfully!")
        print("V2 Enhanced 2DMOINet is ready for use.")
    else:
        print("⚠️  Some pipeline steps failed.")
        print("Please review the errors above and re-run failed steps.")
    
    print("\nPipeline execution completed.")

if __name__ == "__main__":
    main() 