import torch

from dataset_helpers import get_data_loader

num_classes = 8
model_name = 'deeplabv3_mobilenet_v3_large'
weight_backbone = 'MobileNet_V3_Large_Weights.IMAGENET1K_V1'

img_dir_path = './leftImg8bit_trainvaltest/leftImg8bit/'
label_dir_path = './gtFine_trainvaltest/gtFine/'
label_metadata_csv = './labels_meta.csv'
subset = 'val'
batch_size = 2
num_workers = 2
device = torch.device('cuda')

model = torch.load('./cityscapes_seg_model_v1.pt')
model.to(device)

test_dataset, test_loader = get_data_loader(img_dir_path,
                                            label_dir_path,
                                            subset,
                                            label_metadata_csv,
                                            batch_size,
                                            num_workers)

# Check model performance on val set
model.eval()
with torch.no_grad():
    val_acc = 0
    iters = 0
    for imgs, masks in test_loader:
        imgs, masks = imgs.to(device), masks.to(device)

        preds = model(imgs)['out']
        preds = torch.argmax(preds, 1)
        val_acc += (preds == masks).cpu().detach().numpy().mean()
        iters += 1

    print(val_acc / iters)

# 0.8469864152593644
