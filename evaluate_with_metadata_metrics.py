import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from models.conditional_unet import ConditionalUNet
from transition.database.heart_data_loader import PairedImageDataset

# ========== Configuration ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "conditional_unet_metadata.pth"
data_root = "converted_png_top3/testing"
output_dir = "outputs_metadata_eval"
os.makedirs(output_dir, exist_ok=True)

# ========== Load Model ==========
model = ConditionalUNet(img_channels=1, embed_dim=1, hidden_dim=128).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ========== Load Dataset ==========
dataset = PairedImageDataset(root_dir=data_root, image_size=(256, 256), metadata_field="weight")

# ========== Store Metrics ==========
mae_list, ssim_list, psnr_list = [], [], []
names = []

# ========== Inference Loop ==========
for sample in tqdm(dataset, desc="Evaluating"):
    if sample is None:
        continue

    input_img = sample["image_a"].unsqueeze(0).to(device)
    target_img = sample["image_b"].squeeze().cpu().numpy()
    metadata = sample["metadata"].unsqueeze(0).unsqueeze(1).to(device)
    name = sample["name"]
    
    with torch.no_grad():
        pred_img = model(input_img, metadata).squeeze().cpu().numpy()

    # Save prediction
    pred_save_path = os.path.join(output_dir, name)
    pred_pil = Image.fromarray((pred_img * 255).astype(np.uint8))
    pred_pil.save(pred_save_path)

    # Metrics
    if pred_img.shape != target_img.shape:
        continue

    mae = np.mean(np.abs(pred_img - target_img))
    ssim_val = ssim(pred_img, target_img, data_range=1.0)
    psnr_val = psnr(pred_img, target_img, data_range=1.0)

    mae_list.append(mae)
    ssim_list.append(ssim_val)
    psnr_list.append(psnr_val)
    names.append(name)

# ========== Plot Metric ==========
def plot_metric(metric_values, metric_name, save_name):
    plt.figure(figsize=(10, 4))
    plt.plot(metric_values[:20], marker='o')
    plt.title(f"{metric_name} over First 20 Slices")
    plt.xlabel("Slice Index")
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()

plot_metric(mae_list, "MAE", "metric_mae_metadata.png")
plot_metric(ssim_list, "SSIM", "metric_ssim_metadata.png")
plot_metric(psnr_list, "PSNR", "metric_psnr_metadata.png")

print("Evaluation complete. Metrics plots saved.")
