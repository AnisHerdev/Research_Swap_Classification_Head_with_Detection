import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from torchvision import transforms as T
from models.mobilenet_ssd import create_mobilenet_ssd
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

def collate_fn(batch):
    return tuple(zip(*batch))

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
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

def voc_to_coco(val_dataset):
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
def evaluate_model(model, data_loader, device):
    model.eval()
    for idx, (images, targets) in enumerate(tqdm(data_loader, desc="Evaluating")):
        images = [img.to(device) for img in images]
        outputs = model(images)
        for i, (output, target) in enumerate(zip(outputs, targets)):
            pred_labels = output['labels'].cpu().numpy()
            gt_labels = target['labels'].cpu().numpy()
            print(f"Image {idx * data_loader.batch_size + i}: {len(pred_labels)} predictions, {len(gt_labels)} ground truths")
        if idx == 4:
            break  # Only print for first 5 batches
    print("\nDirect label accuracy is not meaningful for detection. Use mAP for proper evaluation.")

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
    for images, targets in tqdm(data_loader, desc="Evaluating"):
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_mobilenet_ssd(num_classes=21).to(device)
    checkpoint = torch.load("checkpoints/ssd_epoch_50.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    val_dataset = get_voc_dataset(root="../VOCdata", year="2012", image_set="val")
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate_fn)
    print(f"Evaluating on {len(val_dataset)} samples...")
    evaluate_model_map(model, val_loader, val_dataset, device)

if __name__ == "__main__":
    main()
