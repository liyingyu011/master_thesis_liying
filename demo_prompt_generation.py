# heart_data_loader.py
import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# Define transformation (only ToTensor if input is numpy array)
transform = transforms.Compose([
    transforms.ToTensor()
])

class PairedImageDataset(Dataset):
    def __init__(self, root_dir, image_size=(256, 256), metadata_field="weight"):
        self.root_dir = root_dir
        self.image_size = image_size
        self.metadata_field = metadata_field

        # Get all patient folders
        self.patient_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                             if os.path.isdir(os.path.join(root_dir, d))]

    def __len__(self):
        return len(self.patient_dirs)

    def _normalize_and_resize(self, image):
        # If image has more than 2 dimensions, extract center slice
        if image.ndim == 3:
            center_slice = image.shape[-1] // 2
            image = image[..., center_slice]
        elif image.ndim > 3:
            image = np.squeeze(image)
            if image.ndim == 3:
                image = image[..., image.shape[-1] // 2]

        # Normalize image
        image = np.clip(image, 0, np.percentile(image, 99))
        image = image / (np.max(image) + 1e-8)

        # Convert to PIL and resize
        image = Image.fromarray((image * 255).astype(np.uint8))
        image = image.resize(self.image_size)

        return np.array(image) / 255.0

    def __getitem__(self, idx):
        patient_dir = self.patient_dirs[idx]
        patient_id = os.path.basename(patient_dir)

        try:
            # Load metadata
            metadata_path = os.path.join(patient_dir, f"{patient_id}_info.cfg")
            meta_dict = self._load_metadata(metadata_path)
            metadata_value = float(meta_dict.get(self.metadata_field, 0.0))

            # Find all frame files
            nii_files = [f for f in os.listdir(patient_dir) if f.endswith(".nii.gz") and "frame" in f and "_gt" not in f]

            # Find frame01
            frame01_file = next((f for f in nii_files if "frame01" in f), None)
            if not frame01_file:
                raise FileNotFoundError("frame01 not found.")

            # Find another frame (frameB)
            other_frames = [f for f in nii_files if f != frame01_file]
            if not other_frames:
                raise FileNotFoundError("No other frame found.")

            def get_frame_num(name):
                return int(name.split("frame")[1].split(".")[0])

            frame01_num = get_frame_num(frame01_file)
            other_frames.sort(key=lambda x: abs(get_frame_num(x) - frame01_num))
            frameB_file = other_frames[0]

            image_a_path = os.path.join(patient_dir, frame01_file)
            image_b_path = os.path.join(patient_dir, frameB_file)

            # Load and process images
            image_a = nib.load(image_a_path).get_fdata()
            image_b = nib.load(image_b_path).get_fdata()

            image_a = self._normalize_and_resize(image_a)
            image_b = self._normalize_and_resize(image_b)

            # Convert to tensor
            image_a = transform(image_a)
            image_b = transform(image_b)

            return {
                "image_a": image_a.float(),
                "image_b": image_b.float(),
                "metadata": torch.tensor(metadata_value, dtype=torch.float)
            }

        except Exception as e:
            print(f"Failed to load data for {patient_id}: {e}")
            return None

    def _load_metadata(self, path):
        meta = {}
        if not os.path.exists(path):
            return meta
        with open(path, "r") as f:
            for line in f:
                if ":" in line:
                    key, val = line.strip().split(":")
                    val = val.strip()
                    try:
                        meta[key.strip()] = float(val) if "." in val else int(val)
                    except:
                        continue
        return meta


# visualize_predictions.py
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def save_visualization(input_img, target_img, predicted_img, idx, save_dir="transition/models/predictions"):
    os.makedirs(save_dir, exist_ok=True)

    def to_numpy(t):
        return t.squeeze().cpu().numpy()

    input_np = to_numpy(input_img)
    target_np = to_numpy(target_img)
    pred_np = to_numpy(predicted_img)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    titles = ["Input", "Target", "Prediction"]
    for ax, img, title in zip(axes, [input_np, target_np, pred_np], titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    save_path = os.path.join(save_dir, f"sample_{idx}.jpg")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")
