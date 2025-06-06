import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
import torchvision.transforms as T
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteHead
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.ssd import SSD
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models.detection.ssd import SSDHead
from torchvision.models.detection._utils import BoxCoder
import torch.nn as nn  # Added import for nn
from tqdm import tqdm
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import os

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]


def build_custom_ssd320(num_classes=21):
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
    base_model = mobilenet_v3_large(weights=weights).features

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
                if idx == 16:
                    features.append(x)
            for layer in self.extras:
                x = layer(x)
                features.append(x)
            return {str(i): f for i, f in enumerate(features)}

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


def get_voc_dataset(root, year="2012", image_set="val"):
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


def collate_fn(batch):
    return tuple(zip(*batch))


@torch.no_grad()
def evaluate_model(model, data_loader, device):
    model.eval()
    print('Evaluating model...')
    for images, targets in tqdm(data_loader):
        images = [img.to(device) for img in images]
        outputs = model(images)
        # Print or process outputs as needed
        print(f"Batch predictions: {[o['labels'].cpu().numpy() for o in outputs]}")
        break  # Remove or adjust this break to process more batches
    print('Evaluation complete. (COCO evaluation logic commented out due to missing imports)')


def voc_to_coco(val_dataset):
    """Convert VOC dataset to COCO format for pycocotools evaluation."""
    images = []
    annotations = []
    categories = []
    ann_id = 1
    for i, cls in enumerate(VOC_CLASSES):
        categories.append({"id": i+1, "name": cls})
    for img_id, (img, target) in enumerate(val_dataset):
        images.append({
            "id": img_id,
            "width": img.shape[2],
            "height": img.shape[1],
            "file_name": str(img_id) + ".jpg"
        })
        boxes = target["boxes"]
        labels = target["labels"]
        for box, label in zip(boxes, labels):
            x_min, y_min, x_max, y_max = box.tolist()
            width = x_max - x_min
            height = y_max - y_min
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(label),
                "bbox": [x_min, y_min, width, height],
                "area": width * height,
                "iscrowd": 0
            })
            ann_id += 1
    return {
        "info": {"description": "VOC to COCO converted dataset"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories
    }


@torch.no_grad()
def evaluate_model_map(model, data_loader, val_dataset, device):
    model.eval()
    print('Evaluating model for mAP...')
    coco_gt_dict = voc_to_coco(val_dataset)
    gt_json = "eval_tmp/voc_gt.json"
    os.makedirs("eval_tmp", exist_ok=True)
    with open(gt_json, "w") as f:
        json.dump(coco_gt_dict, f)
    coco_gt = COCO(gt_json)
    results = []
    img_id = 0
    for images, targets in tqdm(data_loader):
        images = [img.to(device) for img in images]
        outputs = model(images)
        for i, output in enumerate(outputs):
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            for box, score, label in zip(boxes, scores, labels):
                x_min, y_min, x_max, y_max = box.tolist()
                width = x_max - x_min
                height = y_max - y_min
                results.append({
                    "image_id": img_id + i,
                    "category_id": int(label),
                    "bbox": [x_min, y_min, width, height],
                    "score": float(score)
                })
        img_id += len(images)
    pred_json = "eval_tmp/voc_pred.json"
    with open(pred_json, "w") as f:
        json.dump(results, f)
    coco_dt = coco_gt.loadRes(pred_json)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = build_custom_ssd320(num_classes=21).to(device)
    checkpoint = torch.load("checkpoints/ssd_checkpoint_epoch45.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load validation dataset
    val_dataset = get_voc_dataset(root="../VOCdata", year="2012", image_set="val")
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate_fn)

    print(f"Evaluating on {len(val_dataset)} samples...")

    evaluate_model_map(model, val_loader, val_dataset, device)
