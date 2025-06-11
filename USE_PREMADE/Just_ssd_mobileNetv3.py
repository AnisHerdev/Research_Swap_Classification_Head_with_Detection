import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Load image
image_path = 'elephant.jpg'  # <- Replace with your image path
image = Image.open(image_path).convert("RGB")

# Load model and weights
weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = ssdlite320_mobilenet_v3_large(weights=weights)
model.eval()
torch.save(model.state_dict(), "model.pth")

# Preprocess
preprocess = weights.transforms()
img_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

# Inference
with torch.no_grad():
    outputs = model(img_tensor)[0]

# Get COCO labels
labels = outputs['labels']
scores = outputs['scores']
boxes = outputs['boxes']
categories = weights.meta["categories"]

# Threshold for visualization
threshold = 0.5

# Annotate image
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

for label, score, box in zip(labels, scores, boxes):
    if score >= threshold:
        box = box.tolist()
        class_name = categories[label]
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), f"{class_name}: {score:.2f}", fill="yellow", font=font)

# Display the image
plt.imshow(image)
plt.axis('off')
plt.show()
