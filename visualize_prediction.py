import os

import cv2
import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

from dataset_helpers import CityScapes

num_classes = 8
model_name = 'deeplabv3_mobilenet_v3_large'
weight_backbone = 'MobileNet_V3_Large_Weights.IMAGENET1K_V1'

img_dir_path = './leftImg8bit_trainvaltest/leftImg8bit/'
label_dir_path = './gtFine_trainvaltest/gtFine/'
label_metadata_csv = './labels_meta.csv'
subset = 'val'
batch_size = 1
num_workers = 2
device = torch.device('cuda')

categoryId_to_colors = {
    0: [0, 0, 0],
    1: [76, 84, 78],
    2: [142, 100, 82],
    3: [240, 147, 18],
    4: [0, 255, 0],
    5: [135, 206, 235],
    6: [255, 0, 0],
    7: [0, 0, 255]
}

val_transforms = v2.Compose([
    v2.Resize(size=(480, 480), antialias=True),
    v2.PILToTensor(),
    v2.ToDtype(dtype={torch.Tensor: torch.float32, tv_tensors.Mask: torch.int64, "others": None}, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    v2.ToPureTensor()
])
dataset = CityScapes(os.path.join(img_dir_path, subset),
                     os.path.join(label_dir_path, subset),
                     label_metadata_csv,
                     val_transforms)


def label_to_rgb_mask(labels: np.ndarray) -> np.ndarray:
    h, w = labels.shape[:2]
    img_rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for label_id, rgb in categoryId_to_colors.items():
        img_rgb[labels == label_id, :] = rgb
    return img_rgb


model = torch.load('./cityscapes_seg_model_v1.pt')
model.to(device)

# Check model performance on val set
model.eval()
with torch.no_grad():
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    for i in indices:
        img_tensor, _ = dataset[i]
        img_tensor = img_tensor.unsqueeze(0).to(device)

        preds = model(img_tensor)['out']
        preds = torch.argmax(preds, 1).cpu().detach().numpy()[0]

        img, gt_mask = dataset.get_item_for_vis(i)

        output = np.hstack((img, label_to_rgb_mask(gt_mask), label_to_rgb_mask(preds)))
        cv2.imwrite(f'outputs/val_{i}.png', cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
