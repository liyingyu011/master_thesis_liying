# train_metadata_unet.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from heart_data_loader import PairedImageDataset
from conditional_unet import ConditionalUNet

# Define collate function
def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.default_collate(batch)

# Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 100
batch_size = 4
learning_rate = 1e-4

dataset = PairedImageDataset(
    root_dir="transition/database/converted_png",
    image_size=(256, 256),
    metadata_field="Weight"
)


dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_skip_none)

# Model, loss, optimizer
model = ConditionalUNet(img_channels=1, embed_dim=1, hidden_dim=128).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
losses = []
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    valid_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if batch is None:
            continue

        try:
            image_a = batch["image_a"].to(device)
            image_b = batch["image_b"].to(device)
            metadata = batch["metadata"].unsqueeze(1).to(device)

            output = model(image_a, metadata)
            loss = criterion(output, image_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            valid_batches += 1

        except Exception as e:
            print(f"Skipping batch {batch_idx} due to error: {e}")

    avg_loss = epoch_loss / valid_batches if valid_batches > 0 else 0.0
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

# Save model
torch.save(model.state_dict(), "transition/models/conditional_unet_metadata.pth")

# Plot and save loss
plt.plot(losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.grid(True)
plt.savefig("transition/models/loss_curve.png")
plt.show()
