import torch

# Path to the loss file
loss_path = "checkpoints/ssd_checkpoint_loss_epoch130.pt"

# Load the loss history
loss_history = torch.load(loss_path)

# Print the loss values
print("Loss history up to epoch 45:")
for epoch, loss in enumerate(loss_history, 1):
    print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Print the last (most recent) loss value
if loss_history:
    print(f"\nLatest loss (epoch {len(loss_history)}): {loss_history[-1]:.4f}")
else:
    print("No loss history found.")