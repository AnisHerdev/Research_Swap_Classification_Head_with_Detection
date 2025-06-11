import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _mobilenet_extractor
from torchvision.models.detection.ssdlite import SSDLiteHead

# Custom wrapper model
class SSDLiteMobileNetV3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Step 1: Load backbone
        self.backbone, self.out_channels = self.get_mobilenetv3_backbone()

        # Step 2: Anchor generator (adjust sizes/ratios if needed)
        self.anchor_generator = AnchorGenerator(
            sizes=((16,), (32,), (64,), (128,)),
            aspect_ratios=((1.0, 2.0, 0.5),) * 4
        )

        # Step 3: SSDLite detection head
        self.head = SSDLiteHead(
            in_channels=self.out_channels,
            num_anchors=[6] * len(self.out_channels),
            num_classes=num_classes,
            norm_layer=nn.BatchNorm2d
        )

    def get_mobilenetv3_backbone(self):
        backbone = mobilenet_v3_large(weights="IMAGENET1K_V1")

        # Choose which layers to return
        return_layers = {
            "features.4": "0",
            "features.7": "1",
            "features.12": "2",
            "features.16": "3"
        }

        backbone = _mobilenet_extractor(
            backbone,
            trainable_layers=6,
            returned_layers=return_layers,
            fpn=False
        )

        # Output channels for each returned layer
        out_channels = [40, 112, 160, 960]
        return backbone, out_channels

    def forward(self, images):
        if isinstance(images, torch.Tensor):
            images = list(images)  # wrap into list of tensors (like detection API expects)

        # Get feature maps as list
        features_dict = self.backbone(images)
        features_list = list(features_dict.values())

        # Generate anchors
        image_shapes = [img.shape[-2:] for img in images]
        anchors = self.anchor_generator(images, features_list)

        # Get predictions from detection head
        cls_logits, bbox_regression = self.head(features_list)

        return {
            "logits": cls_logits,
            "bbox_regression": bbox_regression,
            "anchors": anchors
        }

# Test run
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SSDLiteMobileNetV3(num_classes=21).to(device)  # e.g., VOC: 20 + background

    dummy_images = torch.randn(2, 3, 300, 300).to(device)

    model.eval()
    with torch.no_grad():
        output = model(dummy_images)

    print("Output keys:", output.keys())
    print("Logits shapes:", [l.shape for l in output["logits"]])
    print("Bbox shapes:", [b.shape for b in output["bbox_regression"]])
    print("Anchors per image:", len(output["anchors"][0]))
