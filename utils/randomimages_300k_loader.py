import numpy as np
import torch


# 300k subset of 80 Million Tiny Images, 300K Random Images, always outlier dataset
class RandomImages(torch.utils.data.Dataset):
    def __init__(self, data_path='/data4/sjma/dataset/OOD/300K_random_images.npy', transform=None):

        self.data = np.load(data_path)
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)

        return img, 0

    def __len__(self):
        return self.data.shape[0]
