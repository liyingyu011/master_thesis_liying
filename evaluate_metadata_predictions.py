import os
import numpy as np
from PIL import Image
from skimage.metrics import mean_squared_error, structural_similarity, peak_signal_noise_ratio
import pandas as pd

# Paths
pred_dir = "inference_outputs_meta"
gt_dir = "medgan_data/val/B"

# Create save path
os.makedirs("evaluation_meta_results", exist_ok=True)

# Collect results
results = []

for fname in sorted(os.listdir(pred_dir)):
    pred_path = os.path.join(pred_dir, fname)
    gt_path = os.path.join(gt_dir, fname)

    if not os.path.exists(gt_path):
        print(f"[Missing] Ground truth not found for: {fname}")
        continue

    pred_img = np.array(Image.open(pred_path).convert('L')).astype(np.float32)
    gt_img = np.array(Image.open(gt_path).convert('L')).astype(np.float32)

    # Normalize to [0, 1]
    pred_img /= 255.0
    gt_img /= 255.0

    mae = np.mean(np.abs(pred_img - gt_img))
    psnr = peak_signal_noise_ratio(gt_img, pred_img, data_range=1.0)
    ssim = structural_similarity(gt_img, pred_img, data_range=1.0)

    results.append({
        "filename": fname,
        "MAE": mae,
        "PSNR": psnr,
        "SSIM": ssim
    })

# Save results
df = pd.DataFrame(results)
df.to_csv("evaluation_meta_results/metrics_summary.csv", index=False)
print("Metrics saved to evaluation_meta_results/metrics_summary.csv")
