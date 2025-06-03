import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

# === 1. Load Pretrained Models ===
ssd_model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)

# === 2. Extract MobileNet Backbone and intermediate features ===
class MobileNetBackbone(nn.Module):
    def __init__(self, mobilenet):
        super().__init__()
        self.features = mobilenet.features
        # Choose layers to extract features from (example: 12 and last)
        self.out_indices = [12, len(mobilenet.features) - 1]
        # Adapt channels if needed
        self.adapt1 = nn.Conv2d(40, 512, 1)
        self.adapt2 = nn.Conv2d(960, 512, 1)

    def forward(self, x):
        features = OrderedDict()
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx == self.out_indices[0]:
                features["0"] = self.adapt1(x)
            if idx == self.out_indices[1]:
                features["1"] = self.adapt2(x)
        return features

backbone = MobileNetBackbone(mobilenet)

# === 3. Use SSD Head from Pretrained SSD ===
ssd_head = ssd_model.head
anchor_generator = ssd_model.anchor_generator
transform = ssd_model.transform

# === 4. Build SSD Model with MobileNet Backbone ===
class SSDMobileNetDetection(nn.Module):
    def __init__(self, backbone, head, anchor_generator, transform):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.anchor_generator = anchor_generator
        self.transform = transform

    def forward(self, images, targets=None):
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        anchors = self.anchor_generator(images, features)
        detections = self.head(features, anchors)
        return detections

model = SSDMobileNetDetection(backbone, ssd_head, anchor_generator, transform)
model.eval()

# === 5. Test with Dummy Input ===
dummy_input = [torch.randn(3, 300, 300)]
with torch.no_grad():
    output = model(dummy_input)
    print("SSD Output:", output)
