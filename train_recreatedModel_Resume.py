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
import os
import matplotlib.pyplot as plt

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

def build_custom_ssd320(num_classes=21):
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
            labels.append(VOC_CLASSES.index(obj['name']) + 1)
        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

    return VOCDetection(
        root=root,
        year=year,
        image_set=image_set,
        download=True,
        transform=lambda img: img,
        target_transform=target_transform
    )

def resume_training(checkpoint_path, resume_from_epoch, more_epochs, checkpoint_basename="ssd_checkpoint"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = build_custom_ssd320(num_classes=21).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.0009, momentum=0.9, weight_decay=5e-4)

    # Load checkpoint
    checkpoint = torch.load(os.path.join("checkpoints", checkpoint_path))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded checkpoint from {checkpoint_path} (epoch {resume_from_epoch})")

    # Load loss history if exists
    loss_file = os.path.join("checkpoints", f"{checkpoint_basename}_loss_epoch{resume_from_epoch}.pt")
    if os.path.exists(loss_file):
        loss_history = torch.load(loss_file)
    else:
        print("Loss history file not found. Starting new loss history.")
        loss_history = []

    voc_dataset = get_voc_dataset(root="./VOCdata", year="2012", image_set="train")
    data_loader = DataLoader(voc_dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_fn)

    for epoch in range(resume_from_epoch, resume_from_epoch + more_epochs):
        model.train()
        running_loss = 0.0
        print(f"\nEpoch {epoch + 1}/{resume_from_epoch + more_epochs}")
        for idx, (images, targets) in enumerate(data_loader):
            print(f"\rProcessing batch {idx + 1}/{len(data_loader)}", end='')
            images = [T.ToTensor()(img).to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()

        avg_loss = running_loss / len(data_loader)
        loss_history.append(avg_loss)
        print(f"\nEpoch [{epoch + 1}], Loss: {avg_loss:.4f}")

        # Save new checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, f"checkpoints/{checkpoint_basename}_epoch{epoch + 1}.pth")
        torch.save(loss_history, f"checkpoints/{checkpoint_basename}_loss_epoch{epoch + 1}.pt")
        print(f"Checkpoint saved at epoch {epoch + 1}")
        torch.cuda.empty_cache()

    # Save new loss curve
    plt.figure()
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Resumed Training Loss Curve")
    plt.grid(True)
    plt.savefig(f"checkpoints/{checkpoint_basename}_loss_curve_resume.png")
    plt.close()
    print(f"Loss curve saved to checkpoints/{checkpoint_basename}_loss_curve_resume.png")


if __name__ == "__main__":
    checkpoint_name = "ssd_checkpoint_epoch100.pth"
    resume_from_epoch = 100
    more_epochs = 30

    resume_training(
        checkpoint_path=checkpoint_name,
        resume_from_epoch=resume_from_epoch,
        more_epochs=more_epochs,
        checkpoint_basename="ssd_checkpoint"
    )
