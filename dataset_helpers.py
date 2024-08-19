import copy
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision import tv_tensors


class CityScapes(Dataset):
    labels_meta_df = None
    id_to_categoryId = None
    id_to_name = None
    categoryId_to_name = None

    def __init__(self, img_dir_path: str,
                 label_dir_path: str,
                 label_metadata_csv: str,
                 transforms=None,
                 img_ext: str = 'png',
                 label_ext: str = 'png'):
        self.img_dir_path = img_dir_path
        self.label_dir_path = label_dir_path
        self.transforms = transforms

        self.sample_paths = []
        for img_fp in glob.iglob(os.path.join(img_dir_path, f'**/*.{img_ext}')):
            # './leftImg8bit_trainvaltest/leftImg8bit/train/zurich/zurich_000015_000019_leftImg8bit.png'

            # ['leftImg8bit_trainvaltest', 'leftImg8bit', 'train', 'zurich', 'zurich_000015_000019_leftImg8bit.png']
            img_fp_comps = os.path.normpath(img_fp).split(os.sep)

            # 'zurich'
            city = img_fp_comps[-2]

            # 'zurich_000015_000019_leftImg8bit'
            img_fn = img_fp_comps[-1].split('.')[0]

            # ['zurich', '000015', '000019']
            label_fn_comps = copy.copy(img_fn).split('_')[:-1]

            # ['zurich', '000015', '000019', 'gtFine', 'labelIds']
            label_fn_comps.extend(['gtFine', 'labelIds'])

            # zurich_000015_000019_gtFine_labelIds
            label_fn = '_'.join(label_fn_comps)

            self.sample_paths.append({
                'img_rel_path': os.sep.join(img_fp_comps[-2:]),
                'mask_rel_path': os.path.join(city, label_fn + f'.{label_ext}')
            })
        CityScapes.labels_meta_df = pd.read_csv(label_metadata_csv)
        CityScapes.id_to_categoryId = pd.Series(CityScapes.labels_meta_df.categoryId.values,
                                                index=CityScapes.labels_meta_df.id).to_dict()
        CityScapes.id_to_name = pd.Series(CityScapes.labels_meta_df.name.values,
                                          index=CityScapes.labels_meta_df.id).to_dict()
        CityScapes.categoryId_to_name = pd.Series(CityScapes.labels_meta_df.category.values,
                                                  index=CityScapes.labels_meta_df.categoryId).to_dict()

    @staticmethod
    def one_hot_encode_mask(mask: np.ndarray) -> torch.Tensor:
        id_mapper = lambda v: CityScapes.id_to_categoryId[v]
        mask_with_categoryId = np.vectorize(id_mapper)(mask)

        # return F.one_hot(torch.from_numpy(mask_with_categoryId))
        return torch.from_numpy(mask_with_categoryId).float()

    def get_item_for_vis(self, idx: int, dims=(480, 480)):
        sample_path = self.sample_paths[idx]
        img = Image.open(os.path.join(self.img_dir_path, sample_path['img_rel_path']))
        mask = Image.open(os.path.join(self.label_dir_path, sample_path['mask_rel_path']))

        img = img.resize(dims, resample=3)  # Bicubic
        mask = mask.resize(dims, resample=0)  # Nearest

        id_mapper = lambda v: CityScapes.id_to_categoryId[v]
        id_to_categoryId_func = np.vectorize(id_mapper)

        mask = id_to_categoryId_func(np.array(mask))
        return img, mask

    def __getitem__(self, idx: int):
        sample_path = self.sample_paths[idx]
        img = Image.open(os.path.join(self.img_dir_path, sample_path['img_rel_path']))
        mask = np.array(Image.open(os.path.join(self.label_dir_path, sample_path['mask_rel_path'])))

        mask = tv_tensors.Mask(CityScapes.one_hot_encode_mask(np.copy(mask)))

        img, mask = self.transforms(img, mask)

        # print(sample_path['img_rel_path'])
        # f, axarr = plt.subplots(1, 2)
        # axarr[0].imshow(img)
        # axarr[1].imshow(mask)
        # plt.show()

        return img, mask

    def __len__(self):
        return len(self.sample_paths)

    def plot_class_histogram(self) -> None:
        counts = {class_id: 0.0 for class_id in CityScapes.categoryId_to_name.keys()}
        id_mapper = lambda v: CityScapes.id_to_categoryId[v]
        id_to_categoryId_func = np.vectorize(id_mapper)

        for i in range(len(self)):
            sample_path = self.sample_paths[i]
            mask = np.array(Image.open(os.path.join(self.label_dir_path, sample_path['mask_rel_path'])))
            mask = id_to_categoryId_func(mask)
            local_counts = dict(zip(*np.unique(mask, return_counts=True)))
            for class_id, count in local_counts.items():
                counts[class_id] += (count / mask.size)
        cateforyName_to_counts = {CityScapes.categoryId_to_name[k]: v / len(self) for k, v in counts.items()}
        print(cateforyName_to_counts)


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
    return batched_imgs


def collate_fn(batch):
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets


def get_data_loader(img_dir_path: str, label_dir_path: str, subset: str,
                    label_metadata_csv: str,
                    batch_size: int = 4,
                    num_workers: int = 2):
    # d = CityScapes(f'./leftImg8bit_trainvaltest/leftImg8bit/{subset}',
    #                f'./gtFine_trainvaltest/gtFine/{subset}/',
    #                './labels_meta.csv', val_transforms)
    if subset == 'train':
        train_transforms = v2.Compose([
            v2.RandomResize(min_size=520, max_size=640, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomCrop(size=(480, 480)),
            v2.PILToTensor(),
            v2.ToDtype(dtype={torch.Tensor: torch.float32, tv_tensors.Mask: torch.int64, "others": None}, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            v2.ToPureTensor()
        ])
        train_dataset = CityScapes(os.path.join(img_dir_path, subset),
                                   os.path.join(label_dir_path, subset),
                                   label_metadata_csv,
                                   train_transforms)
        train_sampler = torch.utils.data.RandomSampler(train_dataset)

        return train_dataset, torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=True,
        )
    elif subset == 'val' or subset == 'test':
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
        sampler = torch.utils.data.SequentialSampler(dataset)

        return dataset, torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, collate_fn=collate_fn
        )


if __name__ == '__main__':
    dataset, _ = get_data_loader('./leftImg8bit_trainvaltest/leftImg8bit/',
                                 './gtFine_trainvaltest/gtFine/',
                                 'train',
                                 './labels_meta.csv')
    dataset[784]

    # img_dir_path = './leftImg8bit_trainvaltest/leftImg8bit/'
    # label_dir_path = './gtFine_trainvaltest/gtFine/'
    # subset = 'val'
    # label_metadata_csv = './labels_meta.csv'
    # dataset = CityScapes(os.path.join(img_dir_path, subset),
    #                      os.path.join(label_dir_path, subset),
    #                      label_metadata_csv,
    #                      None)

    # dataset.plot_class_histogram()
    # train
    # {'void': 0.10248055033323143, 'flat': 0.38833163958637656, 'construction': 0.21911428563735064, 'object': 0.01766535959323915, 'nature': 0.1512629282975397, 'sky': 0.03557920792523552, 'human': 0.011987379378631336, 'vehicle': 0.07357864924839565}
    # val
    # {'void': 0.12024307250976562, 'flat': 0.38111009788513184, 'construction': 0.20560885906219484, 'object': 0.020552252769470213, 'nature': 0.15878013038635255, 'sky': 0.029340121269226076, 'human': 0.013245601654052735, 'vehicle': 0.07111986446380615}
