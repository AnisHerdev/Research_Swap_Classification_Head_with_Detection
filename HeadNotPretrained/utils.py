# Utility functions for metrics, loss logging, etc.
# utils.py
import torch
from torchvision.ops import box_iou

def calculate_mAP(pred_boxes, true_boxes):
    # Placeholder for mAP logic or COCO API integration
    pass

def freeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = False

def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True

def resume_model(model, path, device):
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Model resumed from {path}")

