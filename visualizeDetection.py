import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

# VOC_CLASSES should be defined as a list of 20 class names (no "background")
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

def visualize_predictions(model, dataset, device, num_images=5):
    model.eval()
    for i in range(num_images):
        img, target = dataset[i]
        input_tensor = T.ToTensor()(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)[0]
        plt.figure(figsize=(8, 8))
        img_show = F.to_pil_image(input_tensor.squeeze(0).cpu())
        plt.imshow(img_show)
        # Draw predicted boxes
        for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
            if score > 0.3:  # Show only confident predictions
                xmin, ymin, xmax, ymax = box
                plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                                  fill=False, edgecolor='red', linewidth=2))
                plt.text(xmin, ymin, f"{VOC_CLASSES[label-1]}: {score:.2f}", color='red')
        # Draw ground truth boxes
        for box, label in zip(target['boxes'], target['labels']):
            xmin, ymin, xmax, ymax = box
            plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                              fill=False, edgecolor='green', linewidth=2))
            plt.text(xmin, ymax, f"{VOC_CLASSES[label-1]}", color='green')
        plt.axis('off')
        plt.show()

# Example usage (make sure model, voc_val_dataset, and device are defined in your script):
visualize_predictions(model, voc_val_dataset, device)