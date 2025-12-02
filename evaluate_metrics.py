import torch
import torch.nn as nn
from dataset import MRCTSliceDataset
from train import UNet
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


def evaluate_metrics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = MRCTSliceDataset(task="Task1", region="pelvis", train=False)

    model = UNet().to(device)
    model.load_state_dict(torch.load("checkpoints/unet_mri2ct.pth", map_location=device))
    model.eval()

    psnr_list = []
    ssim_list = []

    with torch.no_grad():
        for i in range(min(100, len(val_dataset))):
            mri, ct_gt = val_dataset[i]
            mri_input = mri.unsqueeze(0).to(device)
            ct_gt_np = ct_gt.squeeze().numpy()

            ct_pred = model(mri_input).squeeze().cpu().numpy()

            psnr_val = psnr(ct_gt_np, ct_pred, data_range=1.0)
            ssim_val = ssim(ct_gt_np, ct_pred, data_range=1.0)

            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)

    print(f"Evaluated {len(psnr_list)} samples")
    print(f"Average PSNR: {np.mean(psnr_list):.2f} dB")
    print(f"Average SSIM: {np.mean(ssim_list):.4f}")

    return psnr_list, ssim_list


if __name__ == "__main__":
    psnr_list, ssim_list = evaluate_metrics()

    # Plot PSNR curve
    plt.figure(figsize=(8, 4))
    plt.plot(psnr_list, label="PSNR", marker='o', linewidth=2)
    plt.axhline(y=25, color='red', linestyle='--', label='Reference 25 dB')
    plt.title("PSNR per Image (Validation Set)", fontsize=14)
    plt.xlabel("Image Index", fontsize=12)
    plt.ylabel("PSNR (dB)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.legend()
    plt.savefig("psnr_curve.png", dpi=300)
    plt.show()

    # Plot SSIM curve
    plt.figure(figsize=(8, 4))
    plt.plot(ssim_list, label="SSIM", marker='s', color="green", linewidth=2)
    plt.axhline(y=0.85, color='red', linestyle='--', label='Reference 0.85')
    plt.title("SSIM per Image (Validation Set)", fontsize=14)
    plt.xlabel("Image Index", fontsize=12)
    plt.ylabel("SSIM", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.legend()
    plt.savefig("ssim_curve.png", dpi=300)
    plt.show()