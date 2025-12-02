
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as T
from dataset import MRCTSliceDataset
from unet import UNet
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "checkpoints/unet_mri2ct.pth"  
instruction = "a small surgical incision in the center"

model = UNet().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

to_pil = T.ToPILImage()

brain_set = MRCTSliceDataset(task="Task1", region="brain", case_id="1BA082")
pelvis_set = MRCTSliceDataset(task="Task1", region="pelvis", case_id="1PA001")

def get_sample_and_prediction(dataset, index):
    mr, ct = dataset[index]
    mr_tensor = mr.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(mr_tensor)[0].cpu()
    return to_pil(mr), to_pil(pred)

brain_input, brain_output = get_sample_and_prediction(brain_set, 5)
pelvis_input, pelvis_output = get_sample_and_prediction(pelvis_set, 5)

fig, ax = plt.subplots(2, 3, figsize=(12, 6))
fig.suptitle("UNet MRI → CT with Natural Language Instruction", fontsize=14)

ax[0, 0].imshow(brain_input, cmap='gray')
ax[0, 0].set_title("Brain MRI Input")
ax[0, 0].axis('off')

ax[0, 1].text(0.5, 0.5, instruction, ha='center', va='center', wrap=True, fontsize=10)
ax[0, 1].set_title("Instruction")
ax[0, 1].axis('off')

ax[0, 2].imshow(brain_output, cmap='gray')
ax[0, 2].set_title("Predicted CT (Brain)")
ax[0, 2].axis('off')

ax[1, 0].imshow(pelvis_input, cmap='gray')
ax[1, 0].set_title("Pelvis MRI Input")
ax[1, 0].axis('off')

ax[1, 1].text(0.5, 0.5, instruction, ha='center', va='center', wrap=True, fontsize=10)
ax[1, 1].set_title("Instruction")
ax[1, 1].axis('off')

ax[1, 2].imshow(pelvis_output, cmap='gray')
ax[1, 2].set_title("Predicted CT (Pelvis)")
ax[1, 2].axis('off')

plt.tight_layout()
plt.savefig("unet_two_region_flowchart.png", dpi=300)
plt.show()
print("image gerenation completion, saved as unet_two_region_flowchart.png")
