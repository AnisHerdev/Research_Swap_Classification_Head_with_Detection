import torch
import torch.nn as nn
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models.detection.backbone_utils import _mobilenet_extractor
from collections import OrderedDict

class MobileNetV3FeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone.features  # Only the feature layers
        # Project 80 channels to 672 channels to match SSDLite's first extra layer
        self.proj = nn.Conv2d(80, 672, kernel_size=1)
        # Add extra layers to match SSDLite's feature map requirements
        self.extra = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(672, 480, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(480, 480, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(480, 512, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=2, stride=2, padding=0),  # 2x2 -> 1x1
                nn.ReLU(inplace=True),
            ),
        ])

    def forward(self, x):
        features = []
        print("\n[DEBUG] Backbone feature extraction:")
        for idx, layer in enumerate(self.backbone):
            x = layer(x)
            print(f"  Layer {idx}: shape={x.shape}")
            # Collect the feature map at idx=10 (80 channels, 20x20)
            if idx == 10:
                x = self.proj(x)
                print(f"  [INFO] Projected feature map at idx={idx} to shape: {x.shape}")
                features.append(x)
                break
        print("\n[DEBUG] Extra layers feature extraction:")
        for i, extra_layer in enumerate(self.extra):
            in_channels = x.shape[1]
            first_conv = extra_layer[0]
            print(f"  Extra layer {i}: input shape={x.shape}, expected in_channels={first_conv.in_channels}, out_channels={first_conv.out_channels}")
            try:
                x = extra_layer(x)
                print(f"    Output shape: {x.shape}")
                features.append(x)
            except Exception as e:
                print(f"    [ERROR] Failed at extra layer {i}: {e}")
                raise
        print(f"\n[DEBUG] Total features collected: {len(features)}")
        for i, f in enumerate(features):
            print(f"  Feature {i}: shape={f.shape}")
        # Return as OrderedDict for compatibility with SSDLite
        keys = ['0', '1', '2', '3', '4', '5']
        return OrderedDict([(k, f) for k, f in zip(keys, features)])

class CustomSSDLite320MobileNetV3:
    def __init__(self, device=None):
        # Set device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # Load classification backbone (pretrained on ImageNet)
        classification_backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        backbone = MobileNetV3FeatureExtractor(classification_backbone)

        # Load detection model (pretrained on COCO)
        detection_model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1)
        detection_head = detection_model.head
        anchor_generator = detection_model.anchor_generator

        # Debug: Print feature map shapes for both backbones
        try:
            backbone.eval()
            detection_model.backbone.eval()
            dummy_input = torch.randn(1, 3, 320, 320)
            with torch.no_grad():
                det_feats = detection_model.backbone(dummy_input)
                custom_feats = backbone(dummy_input)
                # Handle both dict and list outputs
                def get_shapes(feats):
                    if isinstance(feats, dict):
                        return [f.shape for f in feats.values()]
                    return [f.shape for f in feats]
                print("Detection backbone feature shapes:", get_shapes(det_feats))
                print("Custom backbone feature shapes:", get_shapes(custom_feats))
                if get_shapes(det_feats) != get_shapes(custom_feats):
                    print("WARNING: Feature map shapes do not match. This may cause runtime errors.")
        except Exception as e:
            print("Error during feature map shape comparison:",e)

        # Build the model
        model = detection_model
        model.backbone = backbone
        model.head = detection_head
        model.anchor_generator = anchor_generator
        model.to(self.device)
        model.eval()
        self.model = model

    def get_model(self):
        return self.model

# Example usage:
if __name__ == "__main__":
    detector = CustomSSDLite320MobileNetV3()
    model = detector.get_model()
    print("Model loaded successfully.".center( 50, "="))
    dummy_input = torch.randn(1, 3, 320, 320).to(detector.device)
    with torch.no_grad():
        output = model(dummy_input)
    print("Output keys:", output[0])
