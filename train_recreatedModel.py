import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
# import torchvision.transforms.voc as voc_transforms
from torchvision.models.detection.ssd import SSD, SSDHead
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import torchvision.transforms as T
import torch.optim as optim
import os
import matplotlib.pyplot as plt

def build_custom_ssd320(num_classes=21):  # VOC has 20 + background
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
    backbone_model = mobilenet_v3_large(weights=weights)
    backbone = backbone_model.features
    backbone.out_channels = 960

    anchor_generator = DefaultBoxGenerator(
        [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    )

    num_anchors = anchor_generator.num_anchors_per_location()

    ssd_head = SSDHead(
        in_channels=[960],
        num_anchors=num_anchors,
        num_classes=num_classes
    )

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
        head=ssd_head,
        transform=transform
    )

    return model


def collate_fn(batch):
    return tuple(zip(*batch))


def get_voc_dataset(root, year="2012", image_set="train"):
    def target_transform(annotation):
        objs = annotation['annotation']['object']
        if not isinstance(objs, list): objs = [objs]
        boxes = []
        labels = []
        for obj in objs:
            bbox = obj['bndbox']
            box = [
                float(bbox['xmin']),
                float(bbox['ymin']),
                float(bbox['xmax']),
                float(bbox['ymax'])
            ]
            boxes.append(box)
            labels.append(VOC_CLASSES.index(obj['name']) + 1)  # +1 for background class 0
        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

    transforms = T.ToTensor()

    dataset = VOCDetection(
        root=root,
        year=year,
        image_set=image_set,
        download=True,
        transform=lambda img: img,
        target_transform=target_transform
    )

    return dataset


VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

def train(model, data_loader, optimizer, device, num_epochs):
    model.train()
    loss_history = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for idx, (images, targets) in enumerate(data_loader):
            print("\rProcessing batch {}/{}".format(idx + 1, len(data_loader)), end='')
            images = [T.ToTensor()(image).to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()

        avg_loss = running_loss / len(data_loader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        save_path = f"saved_models/ssd_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    # Plot and save loss curve
    os.makedirs("saved_models", exist_ok=True)
    plt.figure()
    plt.plot(range(1, num_epochs + 1), loss_history, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.savefig("saved_models/loss_curve.png")
    plt.close()
    print("Loss curve saved to saved_models/loss_curve.png")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = build_custom_ssd320(num_classes=21).to(device)
    print("Model built successfully.")
    # Load Pascal VOC
    voc_dataset = get_voc_dataset(root="./VOCdata", year="2012", image_set="train")
    data_loader = DataLoader(voc_dataset, batch_size=4, shuffle=True,
                             num_workers=2, collate_fn=collate_fn)
    print(f"Dataset size: {len(voc_dataset)}")
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9, weight_decay=5e-4)

    # Train
    train(model, data_loader, optimizer, device, num_epochs=10)
    save_path = "Final_ssd_mobilenet_v3.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")