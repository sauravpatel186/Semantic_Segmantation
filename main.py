import torchvision
import torch
from torch import nn
from torch.optim.lr_scheduler import PolynomialLR

from dataset_helpers import get_data_loader

device = torch.device('cuda')

num_classes = 8
model_name = 'fcn_resnet50'
weight_backbone = 'ResNet50_Weights.IMAGENET1K_V1'
freeze_until = 156
model_save_path = 'cityscapes_seg_model_v1.pt'

img_dir_path = './leftImg8bit_trainvaltest/leftImg8bit/'
label_dir_path = './gtFine_trainvaltest/gtFine/'
label_metadata_csv = './labels_meta.csv'
batch_size = 2
num_workers = 2

lr = 0.01
momentum = 0.9
wd = 1e-4

epochs = 100


# torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)


def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses["out"]

    return losses["out"] + 0.5 * losses["aux"]


model = torchvision.models.get_model(
    model_name,
    weights=None,
    weights_backbone=weight_backbone,
    num_classes=num_classes,
)

# Freeze some of the initial layers to finetune the model
idx = 0
for name, param in model.named_parameters():
    if idx >= freeze_until:
        break
    # print(idx, name)
    param.requires_grad = False
    idx += 1

print(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)
print(model)
exit()
model.to(device)

train_dataset, train_loader = get_data_loader(img_dir_path,
                                              label_dir_path,
                                              'train',
                                              label_metadata_csv,
                                              batch_size,
                                              num_workers)

val_dataset, val_loader = get_data_loader(img_dir_path,
                                          label_dir_path,
                                          'val',
                                          label_metadata_csv,
                                          batch_size,
                                          num_workers)

params_to_optimize = [
    {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
    {"params": [p for p in model.classifier.parameters() if p.requires_grad]},
]

optimizer = torch.optim.SGD(params_to_optimize, lr=lr, momentum=momentum, weight_decay=wd)

iters_per_epoch = len(train_loader)
lr_scheduler = PolynomialLR(
    optimizer,
    total_iters=iters_per_epoch * epochs, power=0.9
)

# Keep track of min val loss
min_val_loss = 100

# Training loop
for epoch in range(1, epochs + 1):
    train_loss = 0.
    val_loss = 0.

    # Switch model to training mode
    model.train()

    # A pass through train dataset
    for imgs, masks in train_loader:
        # Move batch data to device
        imgs, masks = imgs.to(device), masks.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass the batch and get predictions
        preds = model(imgs)

        # Calculate Loss
        loss = criterion(preds, masks)

        # Add to calculate loss for whole dataset
        train_loss += (loss.item() * imgs.size(0))

        # Backpropagate gradients
        loss.backward()

        # Make weight updates
        optimizer.step()

        lr_scheduler.step()

        # Empty cuda cache to clear useless data from VRAM for better utilization
        torch.cuda.empty_cache()

    # Switch model to inference mode
    model.eval()

    # Check model performance on val set
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            preds = model(imgs)

            loss = criterion(preds, masks)
            val_loss += (loss.item() * imgs.size(0))

            torch.cuda.empty_cache()

    # Calculate avg train and avg val loss
    train_loss /= len(train_dataset)
    val_loss /= len(val_dataset)

    # If loss is decreasing then store model in file else not
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        torch.save(model, model_save_path)

    print('Epoch {}:\tTrain Loss: {}\tVal Loss: {}'.format(epoch, train_loss, val_loss))
