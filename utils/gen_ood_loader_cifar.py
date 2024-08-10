from skimage.filters import gaussian as gblur
from PIL import Image as PILImage
from torchvision import datasets, transforms
import torch
import numpy as np


class AvgOfPair(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.shuffle_indices = np.arange(len(dataset))
        np.random.shuffle(self.shuffle_indices)

    def __getitem__(self, i):
        random_idx = np.random.choice(len(self.dataset))
        while random_idx == i:
            random_idx = np.random.choice(len(self.dataset))

        #return self.dataset[i][0] / 2. + self.dataset[random_idx][0] / 2., 0
        return self.dataset[i][0] / 2. + self.dataset[random_idx][0] / 2., self.dataset[i][1]

    def __len__(self):
        return len(self.dataset)


class GeomMeanOfPair(torch.utils.data.Dataset):
    def __init__(self, dataset, mean, std):
        self.mean = mean
        self.std = std
        self.dataset = dataset
        self.shuffle_indices = np.arange(len(dataset))
        np.random.shuffle(self.shuffle_indices)

    def __getitem__(self, i):
        random_idx = np.random.choice(len(self.dataset))
        while random_idx == i:
            random_idx = np.random.choice(len(self.dataset))

        #return transforms.Normalize(mean, std)(torch.sqrt(self.dataset[i][0] * self.dataset[random_idx][0])), 0
        return transforms.Normalize(self.mean, self.std)(torch.sqrt(self.dataset[i][0] * self.dataset[random_idx][0])), self.dataset[i][1]

    def __len__(self):
        return len(self.dataset)


def gen_ood_data_cifar10(args, type='jigsaw'):
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    ood_data = datasets.CIFAR10(args.data_path, train=True, download=True, transform=test_transform)   # no augmentation
    batch_gen = len(ood_data)
    if type == 'avg-mean':
        print('Generate Average Mean OE!')
        ood_data = datasets.CIFAR10(args.data_path, train=True, download=True, transform=test_transform)   # no augmentation
        ood_loader = torch.utils.data.DataLoader(AvgOfPair(ood_data), batch_size=batch_gen, shuffle=False, num_workers=0)

    elif type == 'geo-mean':
        print('Generate Geometric Mean OE!')
        ood_data = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transforms.ToTensor())   # no augmentation
        ood_loader = torch.utils.data.DataLoader(GeomMeanOfPair(ood_data, mean, std), batch_size=batch_gen, shuffle=False, num_workers=0)

    elif type == 'jigsaw':
        print('Generate jigsaw OE!')
        ood_data = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transforms.ToTensor())   # no augmentation
        ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=batch_gen, shuffle=False, num_workers=0)
        jigsaw = lambda x: torch.cat((
            torch.cat((torch.cat((x[:, 8:16, :16], x[:, :8, :16]), 1),
                        x[:, 16:, :16]), 2),
            torch.cat((x[:, 16:, 16:],
                        torch.cat((x[:, :16, 24:], x[:, :16, 16:24]), 2)), 2),
        ), 1)
        ood_loader.dataset.transform = transforms.Compose([transforms.ToTensor(), jigsaw, transforms.Normalize(mean, std)])

    elif type == 'gauss-speckle':
        print('Generate gauss-speckle OE!')
        ood_data = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transforms.ToTensor())   # no augmentation
        ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=batch_gen, shuffle=False, num_workers=0)
        speckle = lambda x: torch.clamp(x + x * torch.randn_like(x), 0, 1)
        ood_loader.dataset.transform = transforms.Compose([transforms.ToTensor(), speckle, transforms.Normalize(mean, std)])

    elif type == 'uniform-speckle':
        print('Generate uniform-speckle OE!')
        ood_data = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transforms.ToTensor())   # no augmentation
        ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=batch_gen, shuffle=False, num_workers=0)
        speckle = lambda x: torch.clamp(x + x * torch.rand_like(x), 0, 1)
        ood_loader.dataset.transform = transforms.Compose([transforms.ToTensor(), speckle, transforms.Normalize(mean, std)])

    elif type == 'pixelate':
        print('Generate pixelate OE!')
        ood_data = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transforms.ToTensor())   # no augmentation
        ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=batch_gen, shuffle=False, num_workers=0)
        pixelate = lambda x: x.resize((int(32 * 0.2), int(32 * 0.2)), PILImage.Resampling.BOX).resize((32, 32), PILImage.Resampling.BOX)
        ood_loader.dataset.transform = transforms.Compose([pixelate, transforms.ToTensor(), transforms.Normalize(mean, std)])

    elif type == 'rgb-ghost':
        print('Generate rgb ghost OE!')
        ood_data = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transforms.ToTensor())   # no augmentation
        ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=batch_gen, shuffle=False, num_workers=0)
        rgb_shift = lambda x: torch.cat((x[1:2].index_select(2, torch.LongTensor([i for i in range(32 - 1, -1, -1)])),
                                    x[2:, :, :], x[0:1, :, :]), 0)
        ood_loader.dataset.transform = transforms.Compose([transforms.ToTensor(), rgb_shift, transforms.Normalize(mean, std)])

    elif type == 'invert':
        print('Generate invert OE!')
        ood_data = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transforms.ToTensor())   # no augmentation
        ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=batch_gen, shuffle=False, num_workers=0)
        invert = lambda x: torch.cat((x[0:1, :, :], 1 - x[1:2, :, :], 1 - x[2:, :, :],), 0)
        ood_loader.dataset.transform = transforms.Compose([transforms.ToTensor(), invert, transforms.Normalize(mean, std)])

    else:
        print('illegal type!')
        exit()

    return ood_data, ood_loader



def gen_ood_data_cifar100(args, type='jigsaw'):
    mean = [0.5071, 0.4865, 0.4409]
    std = [0.2673, 0.2564, 0.2762]
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    ood_data = datasets.CIFAR100(args.data_path, train=True, download=True, transform=test_transform)   # no augmentation
    batch_gen = len(ood_data)
    if type == 'avg-mean':
        print('Generate Average Mean OE!')
        ood_data = datasets.CIFAR100(args.data_path, train=True, download=True, transform=test_transform)   # no augmentation
        ood_loader = torch.utils.data.DataLoader(AvgOfPair(ood_data), batch_size=batch_gen, shuffle=False, num_workers=0)

    elif type == 'geo-mean':
        print('Generate Geometric Mean OE!')
        ood_data = datasets.CIFAR100(args.data_path, train=True, download=True, transform=transforms.ToTensor())   # no augmentation
        ood_loader = torch.utils.data.DataLoader(GeomMeanOfPair(ood_data, mean, std), batch_size=batch_gen, shuffle=False, num_workers=0)

    elif type == 'jigsaw':
        print('Generate jigsaw OE!')
        ood_data = datasets.CIFAR100(args.data_path, train=True, download=True, transform=transforms.ToTensor())   # no augmentation
        ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=batch_gen, shuffle=False, num_workers=0)
        jigsaw = lambda x: torch.cat((
            torch.cat((torch.cat((x[:, 8:16, :16], x[:, :8, :16]), 1),
                        x[:, 16:, :16]), 2),
            torch.cat((x[:, 16:, 16:],
                        torch.cat((x[:, :16, 24:], x[:, :16, 16:24]), 2)), 2),
        ), 1)
        ood_loader.dataset.transform = transforms.Compose([transforms.ToTensor(), jigsaw, transforms.Normalize(mean, std)])

    elif type == 'gauss-speckle':
        print('Generate gauss-speckle OE!')
        ood_data = datasets.CIFAR100(args.data_path, train=True, download=True, transform=transforms.ToTensor())   # no augmentation
        ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=batch_gen, shuffle=False, num_workers=0)
        speckle = lambda x: torch.clamp(x + x * torch.randn_like(x), 0, 1)
        ood_loader.dataset.transform = transforms.Compose([transforms.ToTensor(), speckle, transforms.Normalize(mean, std)])

    elif type == 'uniform-speckle':
        print('Generate uniform-speckle OE!')
        ood_data = datasets.CIFAR100(args.data_path, train=True, download=True, transform=transforms.ToTensor())   # no augmentation
        ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=batch_gen, shuffle=False, num_workers=0)
        speckle = lambda x: torch.clamp(x + x * torch.rand_like(x), 0, 1)
        ood_loader.dataset.transform = transforms.Compose([transforms.ToTensor(), speckle, transforms.Normalize(mean, std)])

    elif type == 'pixelate':
        print('Generate pixelate OE!')
        ood_data = datasets.CIFAR100(args.data_path, train=True, download=True, transform=transforms.ToTensor())   # no augmentation
        ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=batch_gen, shuffle=False, num_workers=0)
        pixelate = lambda x: x.resize((int(32 * 0.2), int(32 * 0.2)), PILImage.Resampling.BOX).resize((32, 32), PILImage.Resampling.BOX)
        ood_loader.dataset.transform = transforms.Compose([pixelate, transforms.ToTensor(), transforms.Normalize(mean, std)])

    elif type == 'rgb-ghost':
        print('Generate rgb ghost OE!')
        ood_data = datasets.CIFAR100(args.data_path, train=True, download=True, transform=transforms.ToTensor())   # no augmentation
        ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=batch_gen, shuffle=False, num_workers=0)
        rgb_shift = lambda x: torch.cat((x[1:2].index_select(2, torch.LongTensor([i for i in range(32 - 1, -1, -1)])),
                                    x[2:, :, :], x[0:1, :, :]), 0)
        ood_loader.dataset.transform = transforms.Compose([transforms.ToTensor(), rgb_shift, transforms.Normalize(mean, std)])

    elif type == 'invert':
        print('Generate invert OE!')
        ood_data = datasets.CIFAR100(args.data_path, train=True, download=True, transform=transforms.ToTensor())   # no augmentation
        ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=batch_gen, shuffle=False, num_workers=0)
        invert = lambda x: torch.cat((x[0:1, :, :], 1 - x[1:2, :, :], 1 - x[2:, :, :],), 0)
        ood_loader.dataset.transform = transforms.Compose([transforms.ToTensor(), invert, transforms.Normalize(mean, std)])

    else:
        print('illegal type!')
        exit()

    return ood_data, ood_loader
