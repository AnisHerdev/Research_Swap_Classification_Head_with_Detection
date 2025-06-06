import torch
import torchvision
import json
import os
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.transforms import functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from tqdm import tqdm
from torchvision.models.detection.ssd import SSD, SSDHead
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
# from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]
def build_custom_ssd320(num_classes=21):  # 20 VOC classes + background
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
CLASS_TO_IDX = {cls: i + 1 for i, cls in enumerate(VOC_CLASSES)}
IDX_TO_CLASS = {i + 1: cls for i, cls in enumerate(VOC_CLASSES)}


def voc_target_transform(annotation):
    objs = annotation['annotation']['object']
    if not isinstance(objs, list):
        objs = [objs]
    boxes, labels = [], []
    for obj in objs:
        bbox = obj['bndbox']
        box = [
            float(bbox['xmin']), float(bbox['ymin']),
            float(bbox['xmax']), float(bbox['ymax'])
        ]
        boxes.append(box)
        labels.append(CLASS_TO_IDX[obj['name']])
    return {
        'boxes': torch.tensor(boxes, dtype=torch.float32),
        'labels': torch.tensor(labels, dtype=torch.int64)
    }


def get_voc_dataset(root, year="2012", image_set="val"):
    dataset = VOCDetection(
        root=root,
        year=year,
        image_set=image_set,
        download=False,
        transform=lambda img: img,
        target_transform=voc_target_transform
    )
    return dataset


def collate_fn(batch):
    return tuple(zip(*batch))


def evaluate_map(model, dataloader, device):
    model.eval()
    detections = []
    annotation_id = 1

    for i, (images, targets) in enumerate(tqdm(dataloader)):
        images_tensor = [F.to_tensor(img).to(device) for img in images]

        with torch.no_grad():
            outputs = model(images_tensor)

        for image_id, output in enumerate(outputs):
            pred_boxes = output['boxes'].cpu().numpy()
            pred_labels = output['labels'].cpu().numpy()
            pred_scores = output['scores'].cpu().numpy()

            for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                xmin, ymin, xmax, ymax = box
                width = xmax - xmin
                height = ymax - ymin

                detections.append({
                    "image_id": int(i * dataloader.batch_size + image_id),
                    "category_id": int(label),
                    "bbox": [float(xmin), float(ymin), float(width), float(height)],
                    "score": float(score)
                })

    # Save detections to JSON
    os.makedirs("coco_eval", exist_ok=True)
    with open("coco_eval/detections.json", "w") as f:
        json.dump(detections, f)

    # Dummy COCO-style ground truth (you must convert real VOC annotations to COCO for proper eval)
    print("[!] COCO-style ground truth is required for proper mAP evaluation. Use conversion tools like voc2coco.")
    print("[!] Skipping full COCOeval as no real COCO-formatted gt annotations provided.")
    # If you have COCO-style ground truth, load it like this:
    # coco_gt = COCO("path/to/instances_val2012.json")
    # coco_dt = coco_gt.loadRes("coco_eval/detections.json")
    # coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    # coco_eval.evaluate()
    # coco_eval.accumulate()
    # coco_eval.summarize()
    print("Detections saved to coco_eval/detections.json")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # make sure this file exists

    model = build_custom_ssd320(num_classes=21)
    model.load_state_dict(torch.load("checkpoints/final_model_ssd320.pth", map_location=device))
    model.to(device)

    val_dataset = get_voc_dataset("./VOCdata", year="2012", image_set="val")
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    evaluate_map(model, val_loader, device)
