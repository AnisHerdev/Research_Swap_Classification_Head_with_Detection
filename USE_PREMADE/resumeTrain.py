import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from train import get_model, VOCDataset, DetectionTransform, collate_fn, VOC_CLASSES
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import torchvision.transforms as T

def resume_train(checkpoint_path, start_epoch, num_additional_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {checkpoint_path}")

    # Optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=0.0001, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-6
    )
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    # Manually set the learning rate after loading state_dict
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.00001

    # Load previous loss history if available
    loss_history = []
    loss_curve_path = 'checkpoints/training_loss_curve.png'
    if os.path.exists('checkpoints/loss_history.pt'):
        loss_history = torch.load('checkpoints/loss_history.pt')
    else:
        # If not available, fill with zeros up to start_epoch
        loss_history = [0.0] * start_epoch

    # Data augmentation: add random horizontal flip and color jitter
    class AugmentedDetectionTransform(DetectionTransform):
        def __init__(self, size=320):
            super().__init__(size)
            self.aug = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            ])
        def __call__(self, image, target):
            # Apply augmentations before resizing and normalization
            image = self.aug(image)
            return super().__call__(image, target)
    # transform = AugmentedDetectionTransform(size=320)
    transform = DetectionTransform(size=320)
    train_dataset = VOCDataset("../VOCdata", year='2012', image_set='trainval', transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        collate_fn=collate_fn
    )
    print(f"Resuming training from epoch {start_epoch+1} for {num_additional_epochs} more epochs.")
    model.train()
    for epoch in range(start_epoch, start_epoch + num_additional_epochs):
        running_loss = 0.0
        print(f"Epoch {epoch+1}/{start_epoch + num_additional_epochs}")
        for images, targets in tqdm(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            running_loss += losses.item()
        lr_scheduler.step(running_loss)
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}], Loss: {avg_loss:.4f}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        loss_history.append(avg_loss)
        # Save checkpoint
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"ssd_checkpoint_epoch{epoch+1}.pth")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'loss': avg_loss
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        torch.save(loss_history, 'checkpoints/loss_history.pt')
        torch.cuda.empty_cache()
    # Plot and save loss curve
    plt.figure()
    plt.plot(range(1, len(loss_history)+1), loss_history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve (Resumed)')
    plt.grid(True)
    plt.savefig(loss_curve_path)
    print(f'Training loss curve saved as {loss_curve_path}')

if __name__ == '__main__':
    # Set these variables directly instead of using argparse
    checkpoint_path = "checkpoints/ssd_checkpoint_epoch210.pth"  # <-- set your checkpoint path
    start_epoch = 210  # <-- set the epoch value of the checkpoint file
    num_additional_epochs = 20  # <-- set how many more epochs to train
    resume_train(checkpoint_path, start_epoch, num_additional_epochs)
