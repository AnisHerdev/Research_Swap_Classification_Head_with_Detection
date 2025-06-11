import torch
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from pycocotools.coco import COCO
import os

# Set paths
root_dir = "/path/to/coco/val2017"  # folder with images
ann_file = "/path/to/coco/annotations/instances_val2017.json"  # annotation file

# Load model and weights
weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = ssdlite320_mobilenet_v3_large(weights=weights).eval().cuda()

# Preprocessing
transform = weights.transforms()

# Load a subset of COCO (e.g., 100 images)
class CocoSubset(CocoDetection):
    def __init__(self, root, annFile, transform=None, max_imgs=100):
        super().__init__(root, annFile)
        self.ids = self.ids[:max_imgs]  # Limit number of images
        self.transform = transform

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        image_id = self.ids[index]
        if self.transform:
            img = self.transform(img)
        # Add image_id to match COCOEval requirements
        for t in target:
            t["image_id"] = image_id
        return img, target

# Load dataset and data loader
dataset = CocoSubset(root_dir, ann_file, transform=transform, max_imgs=100)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# For COCO evaluation
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.coco_eval import CocoEvaluator
from torchvision.models.detection.coco_utils import get_coco_api_from_dataset

# Run evaluation
coco = get_coco_api_from_dataset(dataset)
evaluator = CocoEvaluator(coco, iou_types=["bbox"])

print("Running evaluation...")
for images, targets in data_loader:
    images = list(img.cuda() for img in images)
    targets = [{k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
    
    with torch.no_grad():
        outputs = model(images)

    res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
    evaluator.update(res)

evaluator.synchronize_between_processes()
evaluator.accumulate()
evaluator.summarize()
