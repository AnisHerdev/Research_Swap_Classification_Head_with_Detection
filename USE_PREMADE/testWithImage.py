import torch
from torchvision import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from train import get_model, VOC_CLASSES

# --------- User variables ---------
image_path = "people.jpeg"  # Set your image path here
checkpoint_path = "checkpoints/ssd_checkpoint_epoch120.pth"  # Set your checkpoint path here
score_threshold = 0.3  # Only show boxes with confidence above this

# --------- Load model ---------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# --------- Image transform (same as train) ---------
transform = T.Compose([
    T.Resize((320, 320)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --------- Load and preprocess image ---------
img = Image.open(image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

# --------- Inference ---------
with torch.no_grad():
    outputs = model(img_tensor)
output = outputs[0]
boxes = output['boxes'].cpu()
labels = output['labels'].cpu()
scores = output['scores'].cpu()

# --------- Visualization ---------
fig, ax = plt.subplots(1)
ax.imshow(img)
for box, label, score in zip(boxes, labels, scores):
    if score < score_threshold:
        # class_name = VOC_CLASSES[label-1] if 0 < label <= len(VOC_CLASSES) else str(label.item())
        # print(f'Skipping box with score {score:.2f} below threshold {score_threshold}, class {class_name}')
        continue
    xmin, ymin, xmax, ymax = box.tolist()
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    class_name = VOC_CLASSES[label-1] if 0 < label <= len(VOC_CLASSES) else str(label.item())
    print(f'Detected {class_name} with score {score:.2f} at [{xmin}, {ymin}, {xmax}, {ymax}]')
    ax.text(xmin, ymin, f'{class_name}: {score:.2f}', color='yellow', fontsize=10, bbox=dict(facecolor='red', alpha=0.5))
plt.axis('off')
plt.show()
