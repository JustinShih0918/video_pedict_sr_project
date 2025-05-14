import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import glob

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now we can import using the full module path
from models.upsampling.super_resolution import CNNUpsampler
from utils.device import get_device

device = get_device()
print(f"Using device: {device}")

# 1. Dataset for your specific directory structure
class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir):
        """
        Initialize dataset with separate directories for low and high resolution images
        
        Args:
            lr_dir: Path to the low-resolution images directory
            hr_dir: Path to the high-resolution images directory
        """
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        
        # Find all subfolders (00001, 00002, etc.)
        self.subfolders = []
        for folder in sorted(os.listdir(lr_dir)):
            if os.path.isdir(os.path.join(lr_dir, folder)) and os.path.isdir(os.path.join(hr_dir, folder)):
                self.subfolders.append(folder)
        
        # Create lists of matching image pairs
        self.lr_files = []
        self.hr_files = []
        
        for folder in self.subfolders:
            lr_subfolder = os.path.join(lr_dir, folder)
            hr_subfolder = os.path.join(hr_dir, folder)
            
            # Recursively find all .png files
            lr_images = sorted(glob.glob(os.path.join(lr_subfolder, "**", "*.png"), recursive=True))
            hr_images = sorted(glob.glob(os.path.join(hr_subfolder, "**", "*.png"), recursive=True))
            
            # Ensure we have matching pairs (same relative paths)
            for lr_path in lr_images:
                rel_path = os.path.relpath(lr_path, lr_subfolder)
                hr_path = os.path.join(hr_subfolder, rel_path)
                if os.path.exists(hr_path):
                    self.lr_files.append(lr_path)
                    self.hr_files.append(hr_path)
        
        print(f"Found {len(self.lr_files)} matching image pairs across {len(self.subfolders)} folders")
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        lr_img = Image.open(self.lr_files[idx])
        hr_img = Image.open(self.hr_files[idx])
        
        # Apply transforms
        return self.transform(lr_img), self.transform(hr_img)

if __name__ == "__main__":
    # 2. Hyperparameters
    batch_size = 16
    epochs = 1
    lr = 1e-4

    # 3. Dataloader with new paths
    lr_dir = 'data/private/train/Low_Resolution'
    hr_dir = 'data/private/train/High_Resolution'
    dataset = SRDataset(lr_dir, hr_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 4. Model
    model = CNNUpsampler(scale=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    # 5. Train loop
    model.train()
    print(f"Starting training for {epochs} epochs...")

    # Create checkpoints directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            
            # Handle potential size mismatch between prediction and ground truth
            if pred.size() != y.size():
                # Crop the larger one to match the smaller one's size
                min_h = min(pred.size(2), y.size(2))
                min_w = min(pred.size(3), y.size(3))
                pred = pred[:, :, :min_h, :min_w]
                y = y[:, :, :min_h, :min_w]
            
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"checkpoints/upsampler_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    # 6. Save final model
    torch.save(model.state_dict(), "checkpoints/upsampler_final.pth")
    print("Training complete. Final model saved to checkpoints/upsampler_final.pth")
