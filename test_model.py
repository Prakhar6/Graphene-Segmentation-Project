#!/usr/bin/env python3
"""
Quick test script to verify the V2 model can be created and run without errors.
"""

import torch
import sys
import os

# Add the models directory to the path
sys.path.append('models/v2_enhanced')

from model_architectures import get_enhanced_model

def test_model():
    print("Testing V2 Enhanced Model...")
    
    try:
        # Create model
        print("Creating model...")
        model = get_enhanced_model(model_type='enhanced', num_classes=4, pretrained=False)
        print("Model created successfully!")
        
        # Test with batch size 4 (matching training config)
        print("Testing forward pass with batch size 4...")
        dummy_input = torch.randn(4, 3, 224, 224)  # Batch size 4, 3 channels, 224x224
        
        # Run forward pass
        with torch.no_grad():
            output = model(dummy_input)
        
        print("Forward pass successful!")
        print(f"Output shape: {output['out'].shape}")
        print(f"Output dtype: {output['out'].dtype}")
        
        # Test with batch size 1 (edge case)
        print("Testing forward pass with batch size 1...")
        dummy_input_single = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            output_single = model(dummy_input_single)
        
        print("Single batch forward pass successful!")
        print(f"Single batch output shape: {output_single['out'].shape}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model()
    if success:
        print("\n✅ Model test passed!")
    else:
        print("\n❌ Model test failed!")
