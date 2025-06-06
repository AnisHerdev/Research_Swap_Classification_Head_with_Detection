# resume_train.py
import torch
import json
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from torchvision.transforms import functional as F
from models.mobilenet_ssd import create_mobilenet_ssd

# VOC class mapping
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
VOC_CLASS_NAME_TO_ID = {name: idx + 1 for idx, name in enumerate(VOC_CLASSES)}  # background = 0

def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform():
    def transform(image, target):
        image = F.to_tensor(image)
        boxes = []
        labels = []
        annotation = target['annotation']
        objs = annotation['object']
        if not isinstance(objs, list):
            objs = [objs]
        for obj in objs:
            name = obj['name']
            label = VOC_CLASS_NAME_TO_ID.get(name, 0)
            bbox = obj['bndbox']
            xmin = float(bbox['xmin'])
            ymin = float(bbox['ymin'])
            xmax = float(bbox['xmax'])
            ymax = float(bbox['ymax'])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
        target_tensor = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }
        return image, target_tensor
    return transform

def prepare_voc_dataset(root='VOCdevkit/', year='2012', image_set='train'):
    dataset = VOCDetection(
        root=root,
        year=year,
        image_set=image_set,
        download=True,
        transforms=get_transform()
    )
    return dataset

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    epoch_loss = 0.0
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        epoch_loss += losses.item()
    avg_loss = epoch_loss / len(data_loader)
    print(f"Epoch [{epoch}] Loss: {avg_loss:.4f}")
    return avg_loss

def plot_losses(losses):
    plt.figure()
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.savefig("loss_plot.png")
    plt.close()

def save_losses(losses, filename="loss_log.json"):
    with open(filename, 'w') as f:
        json.dump(losses, f)

def main(checkpoint_path, resume_epoch, extra_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_mobilenet_ssd(num_classes=21).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    dataset = prepare_voc_dataset(image_set='train')
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    if os.path.exists("loss_log.json"):
        with open("loss_log.json", "r") as f:
            losses = json.load(f)
    else:
        losses = []

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(resume_epoch + 1, resume_epoch + extra_epochs + 1):
        avg_loss = train_one_epoch(model, optimizer, data_loader, device, epoch)
        losses.append(avg_loss)

        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, f"checkpoints/ssd_epoch_{epoch+1}.pth")

        save_losses(losses)
        plot_losses(losses)

if __name__ == "__main__":
    # Set your variables here
    checkpoint_path = "checkpoints/ssd_epoch_4.pth"
    resume_epoch = 4
    extra_epochs = 5

    main(checkpoint_path, resume_epoch, extra_epochs)
