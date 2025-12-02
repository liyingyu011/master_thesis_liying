import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset import MRCTSliceDataset
from train import UNet


def visualize_prediction():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load validation dataset
    val_dataset = MRCTSliceDataset(task="Task1", region="pelvis", train=False)

    # Pick one sample
    mri, ct = val_dataset[0]  # (1, 256, 256)
    mri_input = mri.unsqueeze(0).to(device)  # Add batch dim: (1, 1, H, W)

    # Load model
    model = UNet().to(device)
    model.load_state_dict(torch.load("checkpoints/unet_mri2ct.pth", map_location=device))
    model.eval()

    # Predict
    with torch.no_grad():
        ct_pred = model(mri_input).squeeze().cpu()  # (256, 256)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(mri.squeeze().cpu(), cmap="gray")
    axes[0].set_title("Input MRI")
    axes[1].imshow(ct.squeeze().cpu(), cmap="gray")
    axes[1].set_title("Ground Truth CT")
    axes[2].imshow(ct_pred, cmap="gray")
    axes[2].set_title("Predicted CT")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_prediction()
