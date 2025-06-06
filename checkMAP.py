import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import VOCDetection
from torchvision.transforms import ToTensor
from train_recreatedModel import build_custom_ssd320, get_voc_dataset, collate_fn
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def evaluate_on_voc(model, data_loader, device):
    from torchvision.ops import box_iou
    import numpy as np

    model.eval()
    all_pred_boxes = []
    all_pred_labels = []
    all_pred_scores = []
    all_true_boxes = []
    all_true_labels = []

    with torch.no_grad():
        for idx, (images, targets) in enumerate(data_loader):
            print("\rProcessing batch {}/{}".format(idx + 1, len(data_loader)), end='')
            images = [ToTensor()(img).to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                all_pred_boxes.append(output['boxes'].cpu())
                all_pred_labels.append(output['labels'].cpu())
                all_pred_scores.append(output['scores'].cpu())
                all_true_boxes.append(target['boxes'].cpu())
                all_true_labels.append(target['labels'].cpu())

    # Use torchmetrics for mAP calculation (pip install torchmetrics)
    try:
        metric = MeanAveragePrecision()
        preds = []
        targets_list = []
        for pb, pl, ps, tb, tl in zip(all_pred_boxes, all_pred_labels, all_pred_scores, all_true_boxes, all_true_labels):
            preds.append({'boxes': pb, 'scores': ps, 'labels': pl})
            targets_list.append({'boxes': tb, 'labels': tl})
        metric.update(preds, targets_list)
        results = metric.compute()
        print("mAP:", results['map'].item())
        print("mAP@50:", results['map_50'].item())
        print("mAP@75:", results['map_75'].item())
    except Exception as e:
        print("Error during mAP calculation:", e)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model and checkpoint
    model = build_custom_ssd320(num_classes=21).to(device)
    checkpoint = torch.load("checkpoints/ssd_checkpoint_epoch130.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load VOC validation set
    voc_val_dataset = get_voc_dataset(root="./VOCdata", year="2012", image_set="val")
    val_loader = DataLoader(voc_val_dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn)

    # Evaluate
    evaluate_on_voc(model, val_loader, device)