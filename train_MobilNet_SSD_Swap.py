import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
import torchvision.transforms as T

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
# You must match the number of feature maps and their channels
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
#     print("SSD Output:", output)

# for idx, layer in enumerate(mobilenet.features):
#     x = torch.randn(1, 3, 300, 300)
#     for i in range(idx + 1):
#         x = mobilenet.features[i](x)
#     print(f"Layer {idx}: {x.shape}")

# Helper to convert VOC target to SSD format
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def voc_to_ssd_target(target):
    objs = target['annotation']['object']
    if not isinstance(objs, list):
        objs = [objs]
    boxes = []
    labels = []
    for obj in objs:
        bbox = obj['bndbox']
        xmin = float(bbox['xmin']) - 1
        ymin = float(bbox['ymin']) - 1
        xmax = float(bbox['xmax']) - 1
        ymax = float(bbox['ymax']) - 1
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(VOC_CLASSES.index(obj['name']))
    return {
        'boxes': torch.tensor(boxes, dtype=torch.float32),
        'labels': torch.tensor(labels, dtype=torch.int64)
    }

# Dataset and DataLoader
transform = T.Compose([
    T.Resize((300, 300)),
    T.ToTensor(),
])
train_dataset = VOCDetection(
    root = "./VOCdevkit" ,  # <-- change this to your VOC root
    year='2012',
    image_set='train',
    download=True,
    transform=transform
)
print(len(train_dataset))  # Should print 5717 for VOC2012 train
def collate_fn(batch):
    images, targets = zip(*batch)
    targets = [voc_to_ssd_target(t) for t in targets]
    return list(images), targets

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ssd.to(device)

# Optimizer
optimizer = torch.optim.SGD(ssd.parameters(), lr=0.002, momentum=0.9, weight_decay=5e-4)

# Training loop
num_epochs = 10
ssd.train()
for epoch in range(num_epochs):
    for idx, (images, targets) in enumerate(train_loader):
        print("\r[Batch %d/%d] Processing images..." % (idx + 1, len(train_loader)), end="")
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = ssd(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {losses.item()}")

# Save model
torch.save(ssd.state_dict(), "ssd_mobilenet_custom.pth")