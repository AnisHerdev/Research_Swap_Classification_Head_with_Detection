# Model definition: MobileNetV3 backbone + SSD head
# models/mobilenet_ssd.py
import torch
from torch import nn
from torchvision.models import mobilenet_v3_large
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.ssd import SSD, SSDHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform

def create_mobilenet_ssd(num_classes=21, image_size=320):
    # Load MobileNetV3-Large backbone
    backbone_model = mobilenet_v3_large(weights=None)  # Not pretrained
    backbone = backbone_model.features
    backbone.out_channels = 960

    # Anchor generator for SSD320: 6 feature maps, VOC-style aspect ratios
    anchor_generator = DefaultBoxGenerator(
        aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    )
    num_anchors = anchor_generator.num_anchors_per_location()

    # SSD Head expects a single feature map (last layer)
    head = SSDHead(in_channels=[960], num_anchors=num_anchors, num_classes=num_classes)

    # Image transform
    transform = GeneralizedRCNNTransform(
        min_size=image_size,
        max_size=image_size,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225]
    )

    # Create full model
    model = SSD(
        backbone=backbone,
        anchor_generator=anchor_generator,
        size=(image_size, image_size),
        num_classes=num_classes,
        head=head,
        transform=transform
    )
    return model

