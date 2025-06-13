import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection import _utils as det_utils
from torchvision.datasets import VOCDetection
from torchvision import transforms as T
from torch.utils.data import DataLoader, Subset
from collections import OrderedDict
from functools import partial
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim
import argparse

# --------------------- VOC Classes ---------------------
VOC_CLASSES = [
    'aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
    'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor'
]

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

# --------------------- Data Loading ---------------------
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

# --------------------- Model ---------------------
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

# --------------------- Uncertainty and AL Loop ---------------------
INIT_SIZE = 500
QUERY_SIZE = 250
NUM_ACTIVE_ITER = 10

def compute_uncertainty(model, dataloader, device):
    model.eval()
    uncertainties = []
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc="Uncertainty Estimation")):
            images = [img.to(device) for img in images]
            outputs = model(images)
            for i, output in enumerate(outputs):
                scores = output['scores']
                if len(scores) == 0:
                    uncertainties.append((batch_idx, float('inf')))
                else:
                    probs = F.softmax(scores.unsqueeze(0), dim=1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)
                    mean_entropy = entropy.mean().item()
                    uncertainties.append((batch_idx, mean_entropy))
    return sorted(uncertainties, key=lambda x: -x[1])

def active_learning_loop(full_dataset, device, resume_cycle=0, resume_epoch=0):
    total_indices = list(range(len(full_dataset)))
    random.shuffle(total_indices)
    labeled_indices = total_indices[:INIT_SIZE]
    unlabeled_indices = total_indices[INIT_SIZE:]

    # Try to resume labeled/unlabeled indices if resuming
    if resume_cycle > 0:
        labeled_path = f"saved_state/active_labeled_cycle{resume_cycle}.pt"
        unlabeled_path = f"saved_state/active_unlabeled_cycle{resume_cycle}.pt"
        if os.path.exists(labeled_path) and os.path.exists(unlabeled_path):
            labeled_indices = torch.load(labeled_path)
            unlabeled_indices = torch.load(unlabeled_path)
            print(f"Resumed labeled/unlabeled indices from cycle {resume_cycle}")
        else:
            print(f"Warning: Could not find saved indices for cycle {resume_cycle}, starting from scratch.")

    for cycle in range(resume_cycle, NUM_ACTIVE_ITER):
        print(f"\n=== Active Learning Iteration {cycle+1}/{NUM_ACTIVE_ITER} ===")
        labeled_set = Subset(full_dataset, labeled_indices)
        unlabeled_set = Subset(full_dataset, unlabeled_indices)

        labeled_loader = DataLoader(labeled_set, batch_size=16, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn)
        model = get_model(device)
        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0001, weight_decay=1e-4)

        # Resume model/optimizer if checkpoint exists for this cycle/epoch
        start_epoch = 0
        if cycle == resume_cycle and resume_epoch > 0:
            ckpt_path = f"saved_state/active_model_cycle{cycle+1}_epoch{resume_epoch}.pth"
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = resume_epoch
                print(f"Resumed model/optimizer from {ckpt_path}")
            else:
                print(f"Warning: Could not find checkpoint {ckpt_path}, starting from scratch for this cycle.")

        model.train()
        for epoch in range(start_epoch, 5):
            running_loss = 0
            for images, targets in tqdm(labeled_loader):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {running_loss/len(labeled_loader):.4f}")
        # Save model checkpoint after each cycle (not epoch)
        checkpoint_dir = "saved_state"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"active_model_cycle{cycle+1}.pth")
        torch.save({
            'cycle': cycle+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss/len(labeled_loader)
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        # Select next batch
        unlabeled_loader = DataLoader(unlabeled_set, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)
        uncertainties = compute_uncertainty(model, unlabeled_loader, device)
        selected = [unlabeled_indices[i] for i, _ in uncertainties[:QUERY_SIZE]]

        labeled_indices += selected
        unlabeled_indices = [idx for idx in unlabeled_indices if idx not in selected]
        torch.save(labeled_indices, f"saved_state/active_labeled_cycle{cycle+1}.pt")
        torch.save(unlabeled_indices, f"saved_state/active_unlabeled_cycle{cycle+1}.pt")

    print("\nActive Learning completed!")

# --------------------- Run ---------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Active Learning with Resume Support')
    parser.add_argument('--resume_cycle', type=int, default=0, help='Cycle to resume from (0-based)')
    parser.add_argument('--resume_epoch', type=int, default=0, help='Epoch to resume from (0-based)')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = VOCDataset("../VOCdata", year='2012', image_set='trainval', transform=DetectionTransform())
    active_learning_loop(dataset, device, resume_cycle=args.resume_cycle, resume_epoch=args.resume_epoch)
