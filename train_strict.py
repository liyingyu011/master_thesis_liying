
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MRCTSliceDataset
from unet import UNet
import matplotlib.pyplot as plt

# ===== Config =====
train_path = "data_strict/brain_strict"
val_path = "data_strict/pelvis_strict"
checkpoint_dir = "checkpoints_strict"
os.makedirs(checkpoint_dir, exist_ok=True)

num_epochs = 50
batch_size = 8
lr = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Dataset =====
train_set = MRCTSliceDataset(train_path)
val_set = MRCTSliceDataset(val_path)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# ===== Model =====
model = UNet().to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_loss_log = []
val_loss_log = []

# ===== Evaluation =====
def evaluate(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for mr, ct in loader:
            mr, ct = mr.to(device), ct.to(device)
            output = model(mr)
            loss = criterion(output, ct)
            total_loss += loss.item() * mr.size(0)
    return total_loss / len(loader.dataset)

# ===== Training Loop =====
print("🚀 Starting training on STRICT slices...")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for mr, ct in train_loader:
        mr, ct = mr.to(device), ct.to(device)
        optimizer.zero_grad()
        output = model(mr)
        loss = criterion(output, ct)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * mr.size(0)

    avg_train_loss = epoch_loss / len(train_loader.dataset)
    avg_val_loss = evaluate(model, val_loader)
    train_loss_log.append(avg_train_loss)
    val_loss_log.append(avg_val_loss)

    print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if (epoch + 1) % 10 == 0:
        ckpt_path = os.path.join(checkpoint_dir, f"unet_strict_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)

# Save final model
torch.save(model.state_dict(), os.path.join(checkpoint_dir, "unet_strict_final.pth"))
print("✅ Training complete. Final model saved.")

# Plot loss curve
plt.figure(figsize=(8, 5))
plt.plot(train_loss_log, label="Train Loss")
plt.plot(val_loss_log, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("L1 Loss")
plt.title("Training and Validation Loss (Strict Dataset)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve_strict.png")
plt.show()
