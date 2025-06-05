from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision import transforms
from PIL import Image
import torch

model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
model.eval()

image = Image.open("elephant.jpg").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=model.transform.image_mean, std=model.transform.image_std)
])
input_tensor = [transform(image)]

with torch.no_grad():
    output = model(input_tensor)
print(output)