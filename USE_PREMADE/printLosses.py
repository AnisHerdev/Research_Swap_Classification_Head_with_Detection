import os
import torch
import re

checkpoint_dir = 'checkpoints'  # Change this to your checkpoint directory path

losses = []
def extract_epoch(fname):
    match = re.search(r'epoch(\d+)', fname)
    return int(match.group(1)) if match else -1

files = [fname for fname in os.listdir(checkpoint_dir) if fname.endswith('.pth')]
files.sort(key=extract_epoch)

for fname in files:
    path = os.path.join(checkpoint_dir, fname)
    checkpoint = torch.load(path, map_location='cpu')
    loss = checkpoint.get('loss', None)
    print(f"{fname}: {loss}")
    losses.append(loss)

# Optionally, print all losses as a list
# print("All losses:", losses)