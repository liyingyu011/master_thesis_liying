import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import nibabel as nib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from transition.models.conditional_unet import ConditionalUNet


# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "transition/models/conditional_unet_metadata.pth"
data_dir = "transition/database/testing"
output_path = "transition/models/generated_demo_metadata.png"

# Load model (metadata-based model: embed_dim=1)
model = ConditionalUNet(embed_dim=1)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

# Pick sample patient
patient_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
patient_path = os.path.join(data_dir, patient_dirs[0])  # pick first patient

# Find frame01
nii_files = [f for f in os.listdir(patient_path) if "frame01" in f and f.endswith(".nii.gz")]
if not nii_files:
    raise FileNotFoundError("No frame01.nii.gz file found.")
img_path = os.path.join(patient_path, nii_files[0])

# Load and preprocess image
nii_img = nib.load(img_path).get_fdata()
center_slice = nii_img.shape[-1] // 2 if nii_img.ndim == 3 else 0
img_slice = nii_img[..., center_slice] if center_slice else nii_img
img_slice = np.clip(img_slice, 0, np.percentile(img_slice, 99))
img_slice = img_slice / (np.max(img_slice) + 1e-8)
img_resized = Image.fromarray((img_slice * 255).astype(np.uint8)).resize((256, 256))
img_tensor = torch.tensor(np.array(img_resized) / 255.0).unsqueeze(0).unsqueeze(0).float().to(device)

# Provide metadata input 
metadata = torch.tensor([[70.0]], dtype=torch.float32).to(device)

with torch.no_grad():
    output = model(img_tensor, metadata)
    output_np = output.squeeze().cpu().numpy()

fig, axs = plt.subplots(1, 2, figsize=(6, 3))
axs[0].imshow(img_tensor.squeeze().cpu(), cmap='gray')
axs[0].set_title("Input MRI")
axs[1].imshow(output_np, cmap='gray')
axs[1].set_title("Generated CT")
for ax in axs:
    ax.axis("off")
plt.tight_layout()
output_path = "transition/models/generated_demo_output.jpg"
plt.savefig(output_path)

