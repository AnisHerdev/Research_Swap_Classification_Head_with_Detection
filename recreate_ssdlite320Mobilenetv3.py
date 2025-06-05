import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models.detection.ssd import SSD, SSDHead
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from PIL import Image
from torchvision import transforms


def build_custom_ssd320(num_classes=91):
    # 1. Load pretrained MobileNetV3-Large and extract the feature extractor
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1  # Or use DEFAULT
    backbone_model = mobilenet_v3_large(weights=weights)
    backbone = backbone_model.features
    backbone.out_channels = 960  # Last conv layer output channels

    # 2. Define anchor generator
    anchor_generator = DefaultBoxGenerator(
        [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    )

    # 3. Number of anchors per location
    num_anchors = anchor_generator.num_anchors_per_location()

    # 4. Define SSD head
    ssd_head = SSDHead(
        in_channels=[960],
        num_anchors=num_anchors,
        num_classes=num_classes
    )

    # 5. Define transform
    transform = GeneralizedRCNNTransform(
        min_size=320,
        max_size=320,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225]
    )

    # ❗ FIX: Pass size as a tuple (height, width)
    model = SSD(
        backbone=backbone,
        anchor_generator=anchor_generator,
        size=(320, 320),
        num_classes=num_classes,
        head=ssd_head,
        transform=transform
    )

    return model


# Run and test
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 91
    model = build_custom_ssd320(num_classes).to(device)

    # Replace with your image path
    image_path = "elephant.jpg"
    image = Image.open(image_path).convert("RGB")

    preprocess = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(image).unsqueeze(0).to(device)
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
        print("Detection output:", output)
