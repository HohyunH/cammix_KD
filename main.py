import os
import time
import wandb
import torch
import numpy as np
import torch.nn as nn
import torchvision.models
from dataloader import Intel
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

def save_model(model, saved_dir, file_name='best_model_18.pth'):
    os.makedirs(saved_dir, exist_ok=True)
    check_point = {
        'net': model.state_dict()
    }
    output_path = os.path.join(saved_dir, file_name)
    torch.save(check_point, output_path)

def train(epoch, model, trn_loader, device, criterion, optimizer, path='./model/best_model_18.pth'):
    print("start train...")
    model.train()
    model.to(device)

    trn_loss = 0
    best_loss = np.inf

    for i, (input, target) in enumerate(trn_loader):

        input, target = input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss_ = criterion(output, target)
        loss_.backward()
        optimizer.step()
        trn_loss += loss_

        wandb.log({"Train Loss": loss_})

        _, argmax = torch.max(output, dim=1)
        accuracy = (target == argmax).float().mean()

        wandb.log({"Train Acc": accuracy})

    trn_loss = trn_loss / len(trn_loader)
    print(f"train loss : {trn_loss}")

    if trn_loss < best_loss:
        print('Best performance at epoch: {}, average_loss : {}'.format(epoch + 1, trn_loss))
        best_loss = trn_loss
        # save_model(model, saved_dir)
        torch.save(model.state_dict(), path)

    return trn_loss

def test(model, data_loader, device):
    print("start test...")
    model.eval()
    model.to(device)
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (imgs, labels) in enumerate(data_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            _, argmax = torch.max(outputs, 1)
            total += imgs.size(0)
            correct += (labels==argmax).sum().item()

        print('Test accuracy for {} images: {:.2f}%'.format(total, correct / total * 100))

def main():

    wandb.init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device} is available")

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

    batch_size = 32
    epochs = 10
    learning_rate = 0.001

    trainset = Intel(data_dir="./dataset", mode='train', transform=data_transforms['train'])
    testset = Intel(data_dir="./dataset", mode='test', transform=data_transforms['test'])

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=True)

    model = torchvision.models.resnet18(pretrained=True)
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 6)
    ##
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print(f"------{epoch + 1} epoch in progress------")

        trn_loss = train(epoch, model, trainloader, device, criterion, optimizer, path='./model/best_model_small.pth')
        test(model, testloader, device)


if __name__ == '__main__':
    main()

