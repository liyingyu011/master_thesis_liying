
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MRCTSliceDataset
from unet import UNet
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image

data_root = "data_prepared"
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
instruction = "a small surgical incision in the center"
num_epochs = 50
batch_size = 8
lr = 1e-4

train_set = MRCTSliceDataset(os.path.join(data_root, "brain_val"))
val_set = MRCTSliceDataset(os.path.join(data_root, "pelvis_val"))
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_loss_log = []
val_loss_log = []

def evaluate(model, val_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for mr, ct in val_loader:
            mr, ct = mr.to(device), ct.to(device)
            pred = model(mr)
            loss = criterion(pred, ct)
            total_loss += loss.item() * mr.size(0)
    return total_loss / len(val_loader.dataset)

# ==== training ====
print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for mr, ct in train_loader:
        mr, ct = mr.to(device), ct.to(device)
        optimizer.zero_grad()
        pred = model(mr)
        loss = criterion(pred, ct)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * mr.size(0)

    avg_train_loss = running_loss / len(train_loader.dataset)
    avg_val_loss = evaluate(model, val_loader)

    train_loss_log.append(avg_train_loss)
    val_loss_log.append(avg_val_loss)

    print(f"[Epoch {epoch+1:02d}] Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"unet_epoch{epoch+1}.pth"))

# save model
torch.save(model.state_dict(), os.path.join(checkpoint_dir, "unet_final.pth"))

# ==== visualization ====
plt.figure(figsize=(8, 5))
plt.plot(train_loss_log, label="Train Loss")
plt.plot(val_loss_log, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("L1 Loss")
plt.title("Training & Validation Loss")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.show()

# ==== generatge images ====
model.eval()
to_pil = T.ToPILImage()

mr_tensor, _ = val_set[10]
mr_tensor = mr_tensor.unsqueeze(0).to(device)
with torch.no_grad():
    pred_tensor = model(mr_tensor)[0].cpu()

# save the image
input_img = to_pil(mr_tensor[0].cpu())
output_img = to_pil(pred_tensor)

# ==== flow chart ====
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle("MRI to CT Transformation with Language Instruction")

ax[0].imshow(input_img, cmap='gray')
ax[0].set_title("MRI Input")
ax[0].axis('off')

ax[1].text(0.5, 0.5, instruction, ha='center', va='center', wrap=True, fontsize=12)
ax[1].set_title("Instruction")
ax[1].axis('off')

ax[2].imshow(output_img, cmap='gray')
ax[2].set_title("UNet Output")
ax[2].axis('off')

plt.tight_layout()
plt.savefig("language_driven_flowchart.png", dpi=300)
plt.show()

print("All done: model trained, visualized and instruction-based flowchart saved.")
