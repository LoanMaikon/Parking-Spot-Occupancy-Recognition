from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from glob import glob
from math import ceil
import os
from random import shuffle
import torchvision.transforms as v2
import scipy.io

to_dataset_name = {
    "pklot": "PKLot",
    "cnr": "CNRPark-EXT",
    "plds": "PLds",
}

to_subset_name = {
    "pucpr": "PKLot/PUCPR",
    "ufpr04": "PKLot/UFPR04",
    "ufpr05": "PKLot/UFPR05",
    "camera1": "CNRPark-EXT/CAMERA1",
    "camera2": "CNRPark-EXT/CAMERA2",
    "camera3": "CNRPark-EXT/CAMERA3",
    "camera4": "CNRPark-EXT/CAMERA4",
    "camera5": "CNRPark-EXT/CAMERA5",
    "camera6": "CNRPark-EXT/CAMERA6",
    "camera7": "CNRPark-EXT/CAMERA7",
    "camera8": "CNRPark-EXT/CAMERA8",
    "camera9": "CNRPark-EXT/CAMERA9",
    "isshk": "PLds/ISSHK",
    "vxusd,vmlix": "PLds/VXUSD,VMLIX",
    "qridr": "PLds/QRIDR",
}

class custom_dataset(Dataset):
    def __init__(self, dataset_folder_path, transform, data, first_n_days, balance_data=False):
        self.dataset_folder_path = dataset_folder_path
        self.transform = transform
        self.data = data
        self.first_n_days = first_n_days
        self.balance_data = balance_data

        self.images = []
        self.labels = []

        if len(self.data) != len(self.first_n_days):
            raise ValueError("Length of data must be equal to length of first_n_days.")

        for i in range(len(self.data)):
            n_days = self.first_n_days[i]
            
            subsets = []
            if self.data[i] in to_dataset_name.keys():
                for subset in os.listdir(f"{self.dataset_folder_path}/{to_dataset_name[self.data[i]]}"):
                    subsets.append(to_subset_name[subset.lower()])
            elif self.data[i] in to_subset_name.keys():
                subsets.append(to_subset_name[self.data[i].lower()])
            else:
                raise ValueError(f"Dataset {self.data[i]} not recognized.")
            
            for subset in subsets:
                if n_days == "all":
                    images = glob(f"{self.dataset_folder_path}/{subset}/**/*.jpg", recursive=True)
                    labels = [0 if "empty" in img_path else 1 for img_path in images]
                    self.images.extend(images)
                    self.labels.extend(labels)
                
                elif '>' in n_days or '<=' in n_days:
                    simbol = '>' if '>' in n_days else '<='

                    days = sorted(os.listdir(f"{self.dataset_folder_path}/{subset}"))
                    days_formated = [int("".join(day.split("-"))) for day in days]
                    ordered_days = [day for _, day in sorted(zip(days_formated, days))]
                    threshold_day = int(n_days.replace(simbol, ''))
                    selected_days = ordered_days[threshold_day:] if simbol == '>' else ordered_days[:threshold_day]
                    images = []
                    for day in selected_days:
                        images.extend(glob(f"{self.dataset_folder_path}/{subset}/{day}/**/*.jpg", recursive=True))
                    labels = [0 if "empty" in img_path else 1 for img_path in images]
                    self.images.extend(images)
                    self.labels.extend(labels)

        if self.balance_data:
            self.images, self.labels = self._balance_data(self.images, self.labels)

    def _balance_data(self, images, targets):
        zips = list(zip(images, targets))
        shuffle(zips)
        images, targets = zip(*zips)

        num_targets_1 = 0
        num_targets_0 = 0
        for target in targets:
            if target == 1:
                num_targets_1 += 1
            else:
                num_targets_0 += 1
        
        num_min = min(num_targets_0, num_targets_1)

        new_images = []
        new_targets = []
        targets_1 = 0
        targets_0 = 0
        for i in range(len(images)):
            if targets[i] == 1 and targets_1 < num_min:
                targets_1 += 1
                new_images.append(images[i])
                new_targets.append(targets[i])
            elif targets[i] == 0 and targets_0 < num_min:
                targets_0 += 1
                new_images.append(images[i])
                new_targets.append(targets[i])
        
        return new_images, new_targets
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = read_image(self.images[idx], ImageReadMode.RGB)

        img1 = self.transform(image)
        img2 = self.transform(image)

        return img1, img2
