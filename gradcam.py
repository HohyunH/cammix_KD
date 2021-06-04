#-*- coding: utf-8 -*-

import cv2
import torch
import torchvision
import torchvision.models
import numpy as np
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from dataloader import Intel
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from PIL import Image
import pandas as pd

class Activation_mix():

    def __init__(self, model):
        self.model = model

    def grad_cam(self, img, img_size, img_label):
        self.model.eval()
        hook_outputs = []
        def forward_hook(module, img, output):
            hook_outputs.append(torch.squeeze(output))
        def backward_hook(module, img, output):
            hook_outputs.append(torch.squeeze(output[0]))

        # layer 선택(위 resnet, 아래 simple cnn)
        cam_layer = self.model.layer4[1].conv2
        # cam_layer = net.conv[12]
        cam_layer.register_forward_hook(forward_hook)
        cam_layer.register_backward_hook(backward_hook)
        # print('start gradient CAM')
        hook_outputs = []
        output = self.model(img).squeeze()
        # Ground Truth에 대해서
        # target = img_label
        # 예측에 대해서
        target = F.softmax(output, dim=0).max(0)[1]
        output[target].backward(retain_graph=True)
        # print(f'softmax => {F.softmax(output, dim=0)}, \npredicted class => {target}, ({label_list[target]})')
        a_k = torch.mean(hook_outputs[1], dim=(1, 2), keepdim=True)
        cam_out = torch.sum(a_k * hook_outputs[0], dim=0)
        # normalise
        cam_out = (cam_out + torch.abs(cam_out)) / 2
        cam_out = cam_out / torch.max(cam_out)
        print(cam_out)
        upsampling = torch.nn.Upsample(scale_factor=img_size / len(cam_out), mode='bilinear', align_corners=False)
        resized_cam = upsampling(cam_out.unsqueeze(0).unsqueeze(0)).detach().squeeze().cpu().numpy()
        # print("making cam image")
        # heatmap = cv2.applyColorMap(np.uint8(255 * resized_cam), cv2.COLORMAP_JET)
        # original_img = img.squeeze(dim=0)
        # ## 전처리 과정에서 Normalize 진행시에 다시 복구하는 작업
        # # for j in range(len(mean)):
        # #     original_img[j] *= stdv[j]
        # #     original_img[j] += mean[j]
        # original_img = np.array(original_img.permute(1, 2, 0).cpu() * 255.0)
        # cam_img = heatmap * 0.3 + original_img * 0.5
        # cv2.imwrite('./grad_cam.png', cam_img)
        # cv2.imwrite('./original.png', original_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return resized_cam

    def teacher_cut_mix(self, img, label, cam, cut_length, dataset):
        """
        Perform cutmix by recieving activation map from teacher model
        Increase classification performance by remain most activated part of images
        input data
        * image : original image, 3 channel tensor
        * replace img : added imgae, 3 channel tensor
        * map : activation map, 1 channel numpy array
        * cut length : (width, height), tuple with integer
        * return : np.array image
        """
        # indices = torch.randperm(len(dataset))
        #
        # shuffled_data = dataset[indices]
        # shuffled_target = dataset[indices]

        idx = torch.randint(len(dataset), size=(1,))
        replace_data = dataset[idx][0].unsqueeze(dim=0)
        replace_target = dataset[idx][1]

        width, height = cut_length
        img_size = img.size(-1)
        # Find most activated part from activation map
        # Get index of activated part
        max_col = np.argmax(np.argmax(cam, axis=0))
        max_row = np.argmax(np.argmax(cam, axis=1))

        right = int(max_col + width / 2);
        left = int(max_col - width / 2)
        top = int(max_row + height / 2);
        bottom = int(max_row - height / 2)

        if left < 0:
            right = right - left; left = 0
        if right > img_size:
            right = img_size; left = left + right - img_size

        if bottom < 0:
            top = top - bottom; bottom = 0
        if top > img_size:
            top = img_size; bottom = bottom + top - img_size

        lamb = 1 - ((right - left) * (top - bottom) / (img.size()[-1] * img.size()[-2]))

        try:
            replace_data[:,:,left:right, bottom:top] = img[:,:,left:right, bottom:top]

        except:
            print('Check image size')

        target = (replace_target.unsqueeze(0), label.unsqueeze(0))

        return replace_data, target, lamb

    def feature_map(self, img, img_size):
        vis_model = nn.Sequential(
            *nn.ModuleList(model.children())[:-2]
        )
        heatmap = vis_model(img)
        cam = torch.mean(heatmap, dim=1).squeeze(0)
        upsampling = torch.nn.Upsample(scale_factor=img_size / len(cam), mode='bilinear', align_corners=False)
        resized_cam = upsampling(cam.unsqueeze(0).unsqueeze(0)).detach().squeeze().cpu().numpy()
        return resized_cam


if __name__=="__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])

    model = models.resnet18(pretrained=True)
    model = model.to(device)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 6)
    model.load_state_dict(torch.load("./model/best_model_18.pth"))

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model = model.to(device)

    trainset = Intel(data_dir="./dataset", mode='train', transform=transform)
    testset = Intel(data_dir="./dataset", mode='test', transform=transform)

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, drop_last=True)
    testloader = DataLoader(testset, batch_size=32, shuffle=False, drop_last=True)

    grad = Activation_mix(model)
    input_img = trainset[0][0].unsqueeze(dim=0).to(device)
    img_label = trainset[0][1]
    cam = grad.grad_cam(input_img, 224, img_label)
    print(grad.teacher_cut_mix(input_img, img_label, cam, (100, 100), trainset))

    # cam_result = []
    # for idx in range(len(trainset)):
    #     input_img = trainset[idx][0].unsqueeze(dim=0).to(device)
    #     img_label = trainset[idx][1]
    #     resized_cam = grad_cam(model, input_img, 224, img_label)
    #     saved = np.array(resized_cam)
    #     cam_result.append(saved)
    #     if idx%10 ==0:
    #         print(f"{idx+1} clear....")
    # saved_cam = np.array(cam_result)
    # np.save("./saved_cam",saved_cam)
    # # df = pd.DataFrame(saved)
    # # df.to_csv("sample.csv", index=False, header=False)
