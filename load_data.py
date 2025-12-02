import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def load_case(task="Task1", region="pelvis", case_id="1PA001"):
    """
    Load one MRI-CT case from data folder.

    Parameters:
    - task: "Task1" or "Task2"
    - region: "pelvis" or "brain"
    - case_id: e.g., "1PA001"

    Returns:
    - mr_np: MRI image as numpy array
    - ct_np: CT image as numpy array
    """
    base_path = os.path.join("data", task, region, case_id)
    mr_path = os.path.join(base_path, "mr.nii.gz")
    ct_path = os.path.join(base_path, "ct.nii.gz")

    if not os.path.exists(mr_path) or not os.path.exists(ct_path):
        raise FileNotFoundError(f"Missing file in {base_path}")

    mr_img = nib.load(mr_path)
    ct_img = nib.load(ct_path)

    mr_np = mr_img.get_fdata()
    ct_np = ct_img.get_fdata()

    return mr_np, ct_np


def visualize_slices(mr_np, ct_np, slice_idx=None):
    """
    Show MRI and CT slices side by side.
    """
    if slice_idx is None:
        slice_idx = mr_np.shape[2] // 2  # middle slice

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(mr_np[:, :, slice_idx], cmap="gray")
    plt.title("MRI Slice")

    plt.subplot(1, 2, 2)
    plt.imshow(ct_np[:, :, slice_idx], cmap="gray")
    plt.title("CT Slice")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    mr, ct = load_case(task="Task1", region="pelvis", case_id="1PA001")
    visualize_slices(mr, ct)
