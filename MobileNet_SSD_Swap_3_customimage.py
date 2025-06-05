import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision import transforms
from PIL import Image

# 1. Load Pretrained Models
ssd_model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)

# 2. Define a backbone that outputs multiple feature maps
class MobileNetSSDBackbone(nn.Module):
    def __init__(self, mobilenet):
        super().__init__()
        self.features = mobilenet.features
        # Add layer 16 for a deeper feature map
        self.out_indices = [7, 12, 15, 16]
        # Adapters to match SSD head input channels
        self.adapters = nn.ModuleList([
            nn.Conv2d(80, 512, 1),    # after layer 7
            nn.Conv2d(112, 512, 1),   # after layer 12
            nn.Conv2d(160, 512, 1),   # after layer 15
            nn.Conv2d(960, 512, 1),   # after layer 16
        ])

    def forward(self, x):
        features = []
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.out_indices:
                i = self.out_indices.index(idx)
                features.append(self.adapters[i](x))
        return OrderedDict([(str(i), f) for i, f in enumerate(features)])

backbone = MobileNetSSDBackbone(mobilenet)

# 3. Use SSD's anchor generator and head, but with new feature map shapes
anchor_generator = DefaultBoxGenerator(
    aspect_ratios=[[2, .5], [2, .5, 3, 1/3], [2, .5, 3, 1/3], [2, .5]],
    min_ratio=20,
    max_ratio=90
)

# 4. Build the SSD model using torchvision's SSD class
num_classes = 91  # COCO classes, change if needed
ssd = SSD(
    backbone,
    anchor_generator,
    size=(300, 300),
    num_classes=num_classes,
    head=None,  # Let SSD build the head for you
    image_mean=ssd_model.transform.image_mean,
    image_std=ssd_model.transform.image_std
)

# 5. Test with dummy input
ssd.eval()
# dummy_input = [torch.randn(3, 300, 300)]
# with torch.no_grad():
#     output = ssd(dummy_input)
#     print("SSD Output (dummy):", output)

# 6. Process and test with an actual image
def process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=ssd_model.transform.image_mean, std=ssd_model.transform.image_std)
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image)

# Example usage: replace 'your_image.jpg' with your actual image path
image_path = 'elephant.jpg'
try:
    input_image = process_image(image_path)
    input_tensor = [input_image]
    with torch.no_grad():
        output = ssd(input_tensor)
        print("SSD Output (actual image):", output)
except FileNotFoundError:
    print(f"Image file '{image_path}' not found. Please provide a valid image path.")

# for idx, layer in enumerate(mobilenet.features):
#     x = torch.randn(1, 3, 300, 300)
#     for i in range(idx + 1):
#         x = mobilenet.features[i](x)
#     print(f"Layer {idx}: {x.shape}")