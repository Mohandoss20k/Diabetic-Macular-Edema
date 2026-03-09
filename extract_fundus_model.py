import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import os
import traceback  # Add this import

print("="*60)
print("EXTRACTING FUNDUS MODEL FROM SAVED CHECKPOINT")
print("="*60)

# Define the Fundus model architecture
class FundusModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        base = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(960, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load the saved model
print("\n1. Loading saved model file...")
try:
    checkpoint = torch.load('best_model_7class_fixed.pth', map_location='cpu')
    print(f"Checkpoint type: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Get the state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("Found 'model_state_dict' in checkpoint")
        else:
            state_dict = checkpoint
            print("Using checkpoint directly as state_dict")
        
        print(f"Total keys in state_dict: {len(state_dict.keys())}")
        
        # Create new Fundus model
        print("\n2. Creating Fundus model...")
        fundus_model = FundusModel(num_classes=7)
        
        # Filter state dict to only include Fundus model keys
        print("\n3. Extracting Fundus model weights...")
        fundus_state_dict = {}
        
        # First, look for direct classifier keys (these are the final classifier layers)
        classifier_keys = ['classifier.0.weight', 'classifier.0.bias', 
                          'classifier.1.weight', 'classifier.1.bias', 
                          'classifier.1.running_mean', 'classifier.1.running_var',
                          'classifier.1.num_batches_tracked',
                          'classifier.4.weight', 'classifier.4.bias',
                          'classifier.5.weight', 'classifier.5.bias',
                          'classifier.5.running_mean', 'classifier.5.running_var',
                          'classifier.5.num_batches_tracked',
                          'classifier.8.weight', 'classifier.8.bias']
        
        for key in classifier_keys:
            if key in state_dict:
                fundus_state_dict[key] = state_dict[key]
                print(f"  Found classifier key: {key}")
        
        # Then look for feature extractor keys
        for key in state_dict.keys():
            # Skip if already added
            if key in fundus_state_dict:
                continue
                
            # Look for feature extractor keys (these should start with 'features.' or be in cnn structure)
            if key.startswith('cnn.features.'):
                # Remove 'cnn.' prefix
                new_key = key[4:]
                if new_key not in fundus_state_dict:
                    fundus_state_dict[new_key] = state_dict[key]
                    print(f"  Found feature key: {key} -> {new_key}")
            elif key.startswith('features.'):
                if key not in fundus_state_dict:
                    fundus_state_dict[key] = state_dict[key]
                    print(f"  Found feature key: {key}")
            elif key.startswith('cnn.cnn.features.'):
                # Remove 'cnn.cnn.' prefix
                new_key = key[8:]
                if new_key not in fundus_state_dict:
                    fundus_state_dict[new_key] = state_dict[key]
                    print(f"  Found nested feature key: {key} -> {new_key}")
        
        print(f"\nExtracted {len(fundus_state_dict)} keys for Fundus model")
        
        # Try to load the state dict with strict=False to allow missing keys
        try:
            fundus_model.load_state_dict(fundus_state_dict, strict=False)
            print("✓ Successfully loaded weights into Fundus model")
            
            # Set to evaluation mode before testing
            fundus_model.eval()
            
            # Test inference
            dummy_input = torch.randn(1, 3, 128, 128)
            with torch.no_grad():
                output = fundus_model(dummy_input)
                print(f"✓ Test inference successful. Output shape: {output.shape}")
                print(f"✓ Output values: {output}")
            
            # Save the extracted model
            print("\n4. Saving extracted Fundus model...")
            torch.save({
                'model_state_dict': fundus_model.state_dict(),
                'num_classes': 7,
                'classes': ['No DR signs', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 
                           'Very Severe NPDR', 'PDR', 'Advanced PDR']
            }, 'fundus_model_extracted.pth')
            print("✓ Saved as 'fundus_model_extracted.pth'")
            
        except Exception as e:
            print(f"Error loading weights: {e}")
            traceback.print_exc()
            
    else:
        print(f"Unexpected checkpoint type: {type(checkpoint)}")
        
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()

print("\n" + "="*60)
print("EXTRACTION COMPLETE")
print("="*60)