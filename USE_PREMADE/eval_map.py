import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from torchvision import transforms as T
from train import get_model, VOCDataset, collate_fn, VOC_CLASSES
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
# Use the same transform as in train.py
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

def evaluate_on_voc(model, data_loader, device):
    model.eval()
    metric = MeanAveragePrecision()
    with torch.no_grad():
        for images, targets in tqdm(data_loader):
            images = [img.to(device) for img in images]
            outputs = model(images)
            preds = []
            targets_list = []
            for output, target in zip(outputs, targets):
                preds.append({
                    "boxes": output["boxes"].cpu(),
                    "scores": output["scores"].cpu(),
                    "labels": output["labels"].cpu()
                })
                targets_list.append({
                    "boxes": target["boxes"].cpu(),
                    "labels": target["labels"].cpu()
                })
            metric.update(preds, targets_list)
    results = metric.compute()
    print("mAP:", results['map'].item())
    print("mAP@50:", results['map_50'].item())
    print("mAP@75:", results['map_75'].item())

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load model and checkpoint
    model = get_model(device)
    checkpoint = torch.load("checkpoints_aug/ssd_checkpoint_epoch100.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully.".center(50, "="))
    # Load VOC validation set
    transform = DetectionTransform(size=320)
    voc_val_dataset = VOCDataset("../VOCdata", year="2012", image_set="val", transform=transform)
    val_loader = DataLoader(voc_val_dataset, batch_size=16, shuffle=False, num_workers=16, collate_fn=collate_fn)
    print("Validation dataset loaded successfully.".center(50, "="))
    # Evaluate
    evaluate_on_voc(model, val_loader, device)
