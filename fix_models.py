import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import os

print("="*60)
print("DIABETIC RETINOPATHY MODEL FIX")
print("="*60)

# Define the model architectures (same as in your training code)
class LightweightCNN(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.feat_dim = 960

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.feat_dim, self.feat_dim//8, 1),
            nn.ReLU(),
            nn.Conv2d(self.feat_dim//8, self.feat_dim, 1),
            nn.Sigmoid()
        )

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x * self.channel_attn(x)

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        x = x * self.spatial_attn(spatial)

        x = F.adaptive_avg_pool2d(x, 1)
        return x.view(x.size(0), -1)

class LightweightViT(nn.Module):
    def __init__(self, img_size=128, patch=16, embed_dim=256, depth=6, heads=4):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch, stride=patch)
        self.num_patches = (img_size//patch)**2
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim)*0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim*4,
            dropout=0.2,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.norm(x)

class OCTFusionModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.cnn = LightweightCNN()
        self.vit = LightweightViT()

        self.classifier = nn.Sequential(
            nn.Linear(960+256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        cnn_feat = self.cnn(x)
        vit_feat = self.vit(x)
        combined = torch.cat([cnn_feat, vit_feat], dim=1)
        return self.classifier(combined)

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

print("\n1. CHECKING CURRENT DIRECTORY:")
print("-"*40)
print(f"Current directory: {os.getcwd()}")
print("\nFiles in directory:")
for file in os.listdir():
    if file.endswith('.pth'):
        size = os.path.getsize(file) / (1024*1024)  # Size in MB
        print(f"  - {file} ({size:.2f} MB)")

print("\n2. ANALYZING MODEL FILES:")
print("-"*40)

# Check OCT model
print("\nOCT Model (best_model_5class.pth):")
try:
    oct_data = torch.load('best_model_5class.pth', map_location='cpu')
    print(f"  Type: {type(oct_data)}")
    
    if isinstance(oct_data, dict):
        print(f"  Keys: {list(oct_data.keys())}")
        
        # Try to load into model
        oct_model = OCTFusionModel(num_classes=5)
        
        if 'model_state_dict' in oct_data:
            oct_model.load_state_dict(oct_data['model_state_dict'])
            print("  ✓ Successfully loaded from 'model_state_dict'")
        else:
            oct_model.load_state_dict(oct_data)
            print("  ✓ Successfully loaded direct state dict")
        
        # Test inference
        dummy = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            out = oct_model(dummy)
            print(f"  ✓ Test inference successful. Output shape: {out.shape}")
            
        # Save in a consistent format
        torch.save({
            'model_state_dict': oct_model.state_dict(),
            'num_classes': 5,
            'classes': ['Mild', 'Moderate', 'Normal', 'Proliferate', 'Severe']
        }, 'best_model_5class_fixed.pth')
        print("  ✓ Saved fixed version as 'best_model_5class_fixed.pth'")
            
    else:
        print(f"  ✗ Unexpected format: {type(oct_data)}")
        
except Exception as e:
    print(f"  ✗ Error: {e}")

# Check Fundus model
print("\nFundus Model (best_model_7class.pth):")
try:
    fundus_data = torch.load('best_model_7class.pth', map_location='cpu')
    print(f"  Type: {type(fundus_data)}")
    
    if isinstance(fundus_data, dict):
        print(f"  Keys: {list(fundus_data.keys())}")
        
        # Try to load into model
        fundus_model = FundusModel(num_classes=7)
        
        if 'model_state_dict' in fundus_data:
            fundus_model.load_state_dict(fundus_data['model_state_dict'])
            print("  ✓ Successfully loaded from 'model_state_dict'")
        else:
            fundus_model.load_state_dict(fundus_data)
            print("  ✓ Successfully loaded direct state dict")
        
        # Test inference
        dummy = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            out = fundus_model(dummy)
            print(f"  ✓ Test inference successful. Output shape: {out.shape}")
            
        # Save in a consistent format
        torch.save({
            'model_state_dict': fundus_model.state_dict(),
            'num_classes': 7,
            'classes': ['No DR signs', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 
                       'Very Severe NPDR', 'PDR', 'Advanced PDR']
        }, 'best_model_7class_fixed.pth')
        print("  ✓ Saved fixed version as 'best_model_7class_fixed.pth'")
            
    else:
        print(f"  ✗ Unexpected format: {type(fundus_data)}")
        
except Exception as e:
    print(f"  ✗ Error: {e}")

print("\n" + "="*60)
print("FIX COMPLETE")
print("="*60)
print("\nNow use these files in your Flask app:")
print("  - best_model_5class_fixed.pth (for OCT)")
print("  - best_model_7class_fixed.pth (for Fundus)")