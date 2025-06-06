import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from torchvision.models.detection.ssd import SSD, SSDHead
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import torchvision.transforms as T
import torch.optim as optim
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

def build_custom_ssd320(num_classes=21):
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
    base_model = mobilenet_v3_large(weights=weights).features

    # Define extra layers explicitly
    extra = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(960, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        ),
        nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        ),
        nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        ),
        nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
    ])

    class SSDMobileNetBackbone(nn.Module):
        def __init__(self, base, extras):
            super().__init__()
            self.base = base
            self.extras = extras

        def forward(self, x):
            features = []
            for idx, layer in enumerate(self.base):
                x = layer(x)
                if idx == 16:  # MobileNetV3 feature layer after last conv (960 channels)
                    features.append(x)
            for layer in self.extras:
                x = layer(x)
                features.append(x)
            return {str(i): f for i, f in enumerate(features)}  # return as dict
  # total 5 feature maps

    backbone = SSDMobileNetBackbone(base_model, extra)
    out_channels = [960, 512, 256, 256, 128]

    anchor_generator = DefaultBoxGenerator(
        [[2], [2, 3], [2, 3], [2, 3], [2]]
    )
    num_anchors = anchor_generator.num_anchors_per_location()

    head = SSDHead(out_channels, num_anchors, num_classes)

    transform = GeneralizedRCNNTransform(
        min_size=320,
        max_size=320,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225]
    )

    model = SSD(
        backbone=backbone,
        anchor_generator=anchor_generator,
        size=(320, 320),
        num_classes=num_classes,
        head=head,
        transform=transform
    )
    return model



def collate_fn(batch):
    return tuple(zip(*batch))


def get_voc_dataset(root, year="2012", image_set="train"):
    def target_transform(annotation):
        objs = annotation['annotation']['object']
        if not isinstance(objs, list): objs = [objs]
        boxes, labels = [], []
        for obj in objs:
            bbox = obj['bndbox']
            box = [
                float(bbox['xmin']),
                float(bbox['ymin']),
                float(bbox['xmax']),
                float(bbox['ymax'])
            ]
            boxes.append(box)
            labels.append(VOC_CLASSES.index(obj['name']) + 1)  # +1 for background
        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

    transform = T.Compose([
        T.Resize((320, 320)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    dataset = VOCDetection(
        root=root,
        year=year,
        image_set=image_set,
        download=True,
        transform=transform,
        target_transform=target_transform
    )

    return dataset


def train(model, data_loader, optimizer, scheduler, device, num_epochs, checkpoint_basename="ssd_checkpoint"):
    model.train()
    loss_history = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        for images, targets in tqdm(data_loader):
            # print(f"\rProcessing batch {idx + 1}/{len(data_loader)}", end='')
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()

            running_loss += losses.item()

        scheduler.step()
        avg_loss = running_loss / len(data_loader)
        loss_history.append(avg_loss)
        print(f"\nEpoch [{epoch + 1}], Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, f"checkpoints/{checkpoint_basename}_epoch{epoch + 1}.pth")
        torch.save(loss_history, f"checkpoints/{checkpoint_basename}_loss_epoch{epoch + 1}.pt")
        print(f"Checkpoint saved at epoch {epoch + 1}")
        torch.cuda.empty_cache()

    # Save loss curve
    plt.figure()
    plt.plot(range(1, num_epochs + 1), loss_history, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.savefig(f"checkpoints/{checkpoint_basename}_loss_curve.png")
    plt.close()
    print(f"Loss curve saved to checkpoints/{checkpoint_basename}_loss_curve.png")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = build_custom_ssd320(num_classes=21).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    voc_dataset = get_voc_dataset(root="../VOCdata", year="2012", image_set="train")
    data_loader = DataLoader(voc_dataset, batch_size=8, shuffle=True,
                             num_workers=2, collate_fn=collate_fn)

    print(f"Dataset size: {len(voc_dataset)}")

    train(model, data_loader, optimizer, scheduler, device, num_epochs=45, checkpoint_basename="ssd_checkpoint")

    torch.save(model.state_dict(), "checkpoints/final_model_ssd320.pth")
    print("Final model saved to checkpoints/final_model_ssd320.pth")
