import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection import _utils as det_utils
from torchvision.datasets import VOCDetection
from torchvision import transforms as T
from torch.utils.data import DataLoader
from collections import OrderedDict
from functools import partial
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim

# --------------------- Custom Backbone ---------------------
class MobileNetV3FeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone.features
        self.proj = nn.Conv2d(80, 672, kernel_size=1)
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
                nn.Conv2d(128, 128, kernel_size=2, stride=2, padding=0),
                nn.ReLU(inplace=True),
            ),
        ])

    def forward(self, x):
        features = []
        for idx, layer in enumerate(self.backbone):
            x = layer(x)
            if idx == 10:
                x = self.proj(x)
                features.append(x)
                break
        for extra_layer in self.extra:
            x = extra_layer(x)
            features.append(x)
        keys = ['0', '1', '2', '3', '4', '5']
        return OrderedDict([(k, f) for k, f in zip(keys, features)])

# --------------------- VOC Dataset Setup ---------------------
VOC_CLASSES = [
    'aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
    'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor'
]

class DetectionTransform:
    def __init__(self, size=320):
        self.img_transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __call__(self, image, target):
        image = self.img_transform(image)
        objs = target['annotation']['object']
        if isinstance(objs, dict): objs = [objs]
        boxes, labels = [], []
        for obj in objs:
            bbox = obj['bndbox']
            boxes.append([float(bbox['xmin']), float(bbox['ymin']), float(bbox['xmax']), float(bbox['ymax'])])
            labels.append(VOC_CLASSES.index(obj['name']) + 1)
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }
        return image, target

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root, year='2012', image_set='trainval', transform=None):
        self.dataset = VOCDetection(root, year=year, image_set=image_set, download=True)
        self.transform = transform
    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        if self.transform:
            img, target = self.transform(img, target)
        return img, target
    def __len__(self):
        return len(self.dataset)

def collate_fn(batch):
    return tuple(zip(*batch))

# Data augmentation: add random horizontal flip and color jitter
class AugmentedDetectionTransform(DetectionTransform):
    def __init__(self, size=320):
        super().__init__(size)
        self.aug = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ])
    def __call__(self, image, target):
        image = self.aug(image)
        return super().__call__(image, target)

# --------------------- Load Model ---------------------
def get_model(device):
    classification_backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    backbone = MobileNetV3FeatureExtractor(classification_backbone)

    detection_model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1)
    anchor_generator = detection_model.anchor_generator
    detection_head = detection_model.head

    in_channels = det_utils.retrieve_out_channels(backbone, (320, 320))
    num_anchors = anchor_generator.num_anchors_per_location()
    num_classes = len(VOC_CLASSES) + 1
    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

    detection_head.classification_head = SSDLiteClassificationHead(
        in_channels, num_anchors, num_classes, norm_layer
    )

    model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1)
    model.backbone = backbone
    model.anchor_generator = anchor_generator
    model.head = detection_head
    return model.to(device)

# --------------------- Training Loop ---------------------
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(device)
    print("Device:", device)
    print("Model loaded successfully.".center(50, "="))

    # transform = AugmentedDetectionTransform(size=320)
    transform = DetectionTransform(size=320)
    train_dataset = VOCDataset("../VOCdata", year='2012', image_set='trainval', transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,  # Increase if you have enough GPU memory
        shuffle=True,
        num_workers=16,  # Increase as much as your CPU allows
        pin_memory=True,
        collate_fn=collate_fn
    )
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Number of classes: {len(VOC_CLASSES) + 1}")
    print("Training DataLoader initialized.".center(50, "="))
    params = [p for p in model.parameters() if p.requires_grad]
    # Use AdamW optimizer
    optimizer = optim.AdamW(params, lr=0.0001, weight_decay=1e-4)

    # Use ReduceLROnPlateau scheduler
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-6
    )

    num_epochs = 50
    model.train()

    all_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        print(f"Epoch {epoch+1}/{num_epochs}")
        for images, targets in tqdm(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            running_loss += losses.item()

        lr_scheduler.step(running_loss)
        avg_loss = running_loss/len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        all_losses.append(avg_loss)
        # Save model checkpoint every epoch
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"ssd_checkpoint_epoch{epoch+1}.pth")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'loss': avg_loss
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        torch.cuda.empty_cache()

    # Plot and save loss curve after training
    plt.figure()
    plt.plot(range(1, len(all_losses)+1), all_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.savefig('checkpoints/training_loss_curve.png')
    print('Training loss curve saved as checkpoints/training_loss_curve.png')

if __name__ == '__main__':
    train()