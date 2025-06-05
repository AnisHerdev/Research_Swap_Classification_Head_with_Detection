import torch
import torchvision
import torchvision.transforms.v2 as T
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.ssd import SSDHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.datasets import CocoDetection
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from custom_backbone import MobileNetSSDBackbone  # Assuming your backbone is in this file

# ---- Config ----
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 91  # COCO (1 background + 90 classes)
IMAGE_SIZE = (300, 300)
EPOCHS = 10
BATCH_SIZE = 8
LR = 1e-3

# ---- Backbone ----
mobilenet = torchvision.models.mobilenet_v3_large(weights='IMAGENET1K_V1')
backbone = MobileNetSSDBackbone(mobilenet).to(DEVICE)

# ---- Anchors ----
anchor_generator = DefaultBoxGenerator(
    aspect_ratios=[[2, .5], [2, .5, 3, 1/3], [2, .5, 3, 1/3], [2, .5]],
    min_ratio=20,
    max_ratio=90
)

# ---- Transform ----
transform = GeneralizedRCNNTransform(
    min_size=IMAGE_SIZE[0],
    max_size=IMAGE_SIZE[1],
    image_mean=[0.48235, 0.45882, 0.40784],  # VGG SSD defaults
    image_std=[1.0, 1.0, 1.0]
)

# ---- Model ----
ssd = SSD(
    backbone=backbone,
    anchor_generator=anchor_generator,
    size=IMAGE_SIZE,
    num_classes=NUM_CLASSES,
    image_mean=[0.48235, 0.45882, 0.40784],
    image_std=[1.0, 1.0, 1.0]
).to(DEVICE)

# ---- Dataset ----
def collate_fn(batch):
    return tuple(zip(*batch))

transform_coco = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.ToTensor(),
])

train_dataset = CocoDetection(
    root='path/to/train/images',
    annFile='path/to/annotations/instances_train2017.json',
    transforms=transform_coco
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=4)

# ---- Optimizer ----
params = [p for p in ssd.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4)

# ---- Training Loop ----
for epoch in range(EPOCHS):
    ssd.train()
    running_loss = 0.0
    for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images = list(img.to(DEVICE) for img in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = ssd(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

    print(f"Epoch {epoch+1} Loss: {running_loss/len(train_loader)}")

    # Save model
    torch.save(ssd.state_dict(), f"ssd_mobilenet_epoch{epoch+1}.pth")

# ---- Inference Test (Optional) ----
ssd.eval()
dummy = [torch.randn(3, 300, 300).to(DEVICE)]
with torch.no_grad():
    output = ssd(dummy)
    print("Sample output:", output)
# ---- Save Final Model ----
torch.save(ssd.state_dict(), "ssd_mobilenet_final.pth")