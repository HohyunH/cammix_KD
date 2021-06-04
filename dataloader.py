#-*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
import os


class Intel(Dataset):

    def __init__(self, data_dir, mode, transform=None):
        self.all_data = sorted(glob.glob(os.path.join(data_dir, mode, '*', '*', '*')))
        self.transform = transform

    def __len__(self):
        length = len(self.all_data)
        return  length

    def __getitem__(self, idx):
        data_path = self.all_data[idx]
        img = Image.open(data_path)
        if self.transform is not None:
            img = self.transform(img)

        dir_name = os.path.dirname(data_path)

        class_name = dir_name.split("\\")
        class_name = class_name[-1]
        if class_name == 'buildings':
            label = 0
        elif class_name == 'forest':
            label = 1
        elif class_name == 'glacier':
            label = 2
        elif class_name == 'mountain':
            label = 3
        elif class_name == 'sea':
            label = 4
        else :
            label = 5

        return img, torch.tensor(label)


if __name__ == "__main__":
    print('pytorch version: {}'.format(torch.__version__))
    print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize([120, 120]),
            transforms.ToTensor()
        ]),
        'test': transforms.Compose([
            transforms.Resize([120, 120]),
            transforms.ToTensor()
        ]),
    }

    batch_size = 16

    trainset = Intel(data_dir="./dataset", mode='train', transform=data_transforms['train'])
    testset = Intel(data_dir="./dataset", mode='test', transform=data_transforms['test'])

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last= True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=True)
    for img, label in trainloader:
        print(img)
        print(label[0])

        import sys;
        sys.exit(0)
