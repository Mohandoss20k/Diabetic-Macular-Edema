# check_model_structure.py
import torch
import sys

print("="*60)
print("MODEL STRUCTURE CHECK")
print("="*60)

# Check OCT model
print("\n1. CHECKING OCT MODEL (best_model_5class.pth)")
print("-"*40)
try:
    oct_data = torch.load('best_model_5class.pth', map_location='cpu')
    print(f"Type: {type(oct_data)}")
    
    if isinstance(oct_data, dict):
        print(f"Keys: {list(oct_data.keys())}")
        
        # Check if it's a checkpoint with model_state_dict
        if 'model_state_dict' in oct_data:
            print("✓ Contains 'model_state_dict' key")
            state_dict = oct_data['model_state_dict']
            print(f"  State dict type: {type(state_dict)}")
            print(f"  Number of layers: {len(state_dict.keys())}")
            print(f"  Sample keys: {list(state_dict.keys())[:5]}")
            
            # Check if it contains training history
            if 'train_acc' in oct_data:
                print(f"  Contains training history")
        else:
            print("⚠ Direct state dict (no wrapper)")
            print(f"  Number of layers: {len(oct_data.keys())}")
            print(f"  Sample keys: {list(oct_data.keys())[:5]}")
    else:
        print(f"⚠ Unexpected type: {type(oct_data)}")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Check Fundus model
print("\n2. CHECKING FUNDUS MODEL (best_model_7class.pth)")
print("-"*40)
try:
    fundus_data = torch.load('best_model_7class.pth', map_location='cpu')
    print(f"Type: {type(fundus_data)}")
    
    if isinstance(fundus_data, dict):
        print(f"Keys: {list(fundus_data.keys())}")
        
        # Check if it's a checkpoint with model_state_dict
        if 'model_state_dict' in fundus_data:
            print("✓ Contains 'model_state_dict' key")
            state_dict = fundus_data['model_state_dict']
            print(f"  State dict type: {type(state_dict)}")
            print(f"  Number of layers: {len(state_dict.keys())}")
            print(f"  Sample keys: {list(state_dict.keys())[:5]}")
            
            # Check if it contains validation metrics
            if 'val_f1_macro' in fundus_data:
                print(f"  Validation F1: {fundus_data['val_f1_macro']}")
        else:
            print("⚠ Direct state dict (no wrapper)")
            print(f"  Number of layers: {len(fundus_data.keys())}")
            print(f"  Sample keys: {list(fundus_data.keys())[:5]}")
    else:
        print(f"⚠ Unexpected type: {type(fundus_data)}")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("CHECK COMPLETE")
print("="*60)