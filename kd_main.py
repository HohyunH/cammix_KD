import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from model import SimpleCNN
from dataloader import Intel
from gradcam import Activation_mix

def loss_fn_kd(outputs, labels, teacher_outputs, alpha, T):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    # alpha = params.alpha
    # T = params.temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss


def kd_train(model, teacher_model, train_loader, optimizer, cutmix_prob, alpha, T):

    model.train()
    model.to(device)
    teacher_model.eval()
    teacher_model.to(device)
    best_loss = np.inf
    grad = Activation_mix(model)

    alpha = 0.1
    T = 0.1

    trn_plot = []
    trn_loss = 0
    for i, (input, label) in enumerate(train_loader):

        input, label = input.to(device), label.to(device)
        r = np.random.rand(1)

        if r < cutmix_prob:
            loss = 0
            for input_img, img_label in zip(input, label):
                input_img = input_img.unsqueeze(0)

                input_img, img_label = input_img.to(device), img_label.to(device)

                with torch.no_grad():
                    teacher_output = teacher_model(input_img)

                # cam = grad.grad_cam(input_img, 224, img_label)
                cam = grad.feature_map(input_img, 224)
                replace_data, target, lamb = grad.teacher_cut_mix(input_img, img_label, cam, (100, 100), trainset)

                replace_data = replace_data.to(device)

                r_target = target[0].to(device)
                t_target = target[1].to(device)

                output = model(replace_data)

                cut_loss = loss_fn_kd(output, r_target, teacher_output, alpha, T).to(device) * lamb \
                           + loss_fn_kd(output, t_target, teacher_output, alpha, T).to(device)*(1.-lamb)

                loss+=cut_loss
            loss = loss/len(input)
            # print("using cutmix loss")
            trn_plot.append(loss)
        else:

            input, label = input.to(device), label.to(device)
            with torch.no_grad():
                teacher_outputs = teacher_model(input)

            output = model(input)
            loss = loss_fn_kd(output, label, teacher_outputs, alpha, T)
            # print("using normal loss")
        trn_loss += loss
        trn_plot.append(loss)

        if i%30 == 0:
            print(f"training loss = {loss}")
        trn_loss = trn_loss / input.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if trn_loss < best_loss:
        print('Best performance at epoch: {}, average_loss : {}'.format(e, trn_loss))
        best_loss = trn_loss
        # save_model(model, saved_dir)
        torch.save(model.state_dict(), './model/just_kd_model_18.pth')

    trn_loss = trn_loss / len(train_loader)

    # plt.plot(trn_plot.cpu().numpy())
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



if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default="0.5", help='Input the probability of KD loss')
    parser.add_argument('--T', type=float, default="0.1", help='Input the temperature of softmax probability')
    parser.add_argument('--teacher', type=str, default='resnet50', help='determining the kind of model')
    parser.add_argument('--model', type=str, default='resnet50', help='determining the kind of model')
    parser.add_argument('--cammix_prob', type=float, default='0.5', help='Input the probability of cammix')
    parser.add_argument('--epoch', type=int, default='5', help='Input the number of epoch')
    parser.add_argument('--device', type=str, default='cpu', help='determining the kind of device')

    args = parser.parse_args()

    device = args.device

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(5),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(120),
            transforms.Resize([224, 224]),
            transforms.ToTensor()
        ]),
        'test': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor()
        ]),
    }

    if args.teacher == "resnet18":
        model = models.resnet18(pretrained=True)
    else :
        model = models.resnet50(pretrained=True)
    model = model.to(device)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 6)
    model.load_state_dict(torch.load("./model/best_model_18.pth"))


    if args.model == "resnet18":
        teacher_model = models.resnet18(pretrained=True)
    else :
        teacher_model = models.resnet50(pretrained=True)
    teacher_model = teacher_model.to(device)
    num_ftrs = teacher_model.fc.in_features
    teacher_model.fc = nn.Linear(num_ftrs, 6)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(teacher_model.parameters(), lr=0.001)
    # teacher_model.load_state_dict(torch.load("./model/just_kd_model_18.pth"))

    trainset = Intel(data_dir="./dataset", mode='train', transform=data_transforms['train'])
    testset = Intel(data_dir="./dataset", mode='test', transform=data_transforms['test'])

    trainloader = DataLoader(trainset, batch_size=16, shuffle=True, drop_last=True)
    testloader = DataLoader(testset, batch_size=16, shuffle=False, drop_last=True)

    for e in range(1, args.epoch+1):
        print(f"{e} epochs processing")
        kd_train(model, teacher_model, trainloader, optimizer, args.cammix_prob, args.alpha, args.T)

    print("teacher model")
    test(model, testloader, device)
    print("student model")
    test(teacher_model, testloader, device)