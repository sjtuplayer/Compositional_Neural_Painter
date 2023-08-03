import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import random
import torch.nn.functional as F
# Decide which device we want to run on
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from PIL import Image
from torchvision import transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def read_img(img_path, img_type='RGB', size=128):
    h=w=size
    img = Image.open(img_path).convert(img_type)
    if h is not None and w is not None:
        img = img.resize((w, h), resample=Image.NEAREST)
    img = np.array(img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float() / 255.
    if img.shape[-1]>1024 and img.shape[-2]>1024:
        resize=transforms.Resize((1024,1024))
        img=resize(img)
    return img
class PixelLoss(nn.Module):

    def __init__(self, p=1):
        super(PixelLoss, self).__init__()
        self.p = p

    def forward(self, canvas, gt, ignore_color=True):
        if gt.max()>2:
            canvas=canvas/255.0
            gt=gt/255.0
        if ignore_color:
            canvas = torch.mean(canvas, dim=1)
            gt = torch.mean(gt, dim=1)
        loss = (torch.abs(canvas-gt)).mean(dim=(1,2))
        return loss


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True).to(device)
        blocks = []
        blocks.append(vgg.features[:4].eval())
        blocks.append(vgg.features[4:9].eval())
        blocks.append(vgg.features[9:16].eval())
        blocks.append(vgg.features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        self.resize = resize

    def forward(self, input, target, ignore_color=False):
        self.mean = self.mean.type_as(input)
        self.std = self.std.type_as(input)
        if ignore_color:
            input = torch.mean(input, dim=1, keepdim=True)
            target = torch.mean(target, dim=1, keepdim=True)
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = torch.zeros(input.shape[0]).float().to(device)
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss+=(torch.abs(x-y)).sum(dim=(1,2,3))
            #loss += torch.nn.functional.l1_loss(x, y)
        return loss



class VGGStyleLoss(torch.nn.Module):
    def __init__(self, transfer_mode=1, resize=True):
        super(VGGStyleLoss, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True).to(device)
        for i, layer in enumerate(vgg.features):
            if isinstance(layer, torch.nn.MaxPool2d):
                vgg.features[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        blocks = []
        if transfer_mode == 0:  # transfer color only
            blocks.append(vgg.features[:4].eval())
            blocks.append(vgg.features[4:9].eval())
        else: # transfer both color and texture
            blocks.append(vgg.features[:4].eval())
            blocks.append(vgg.features[4:9].eval())
            blocks.append(vgg.features[9:16].eval())
            blocks.append(vgg.features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)

        self.transform = torch.nn.functional.interpolate
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self.resize = resize

    def gram_matrix(self, y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * w * h)
        return gram

    def forward(self, input, target):
        if target.max()>2:
            input=input/255.0
            target=target/255.0
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = torch.zeros(input.shape[0]).float().to(device)
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            gm_x = self.gram_matrix(x)
            gm_y = self.gram_matrix(y)
            loss += ((gm_x-gm_y)**2).sum(dim=(1,2))
        return loss
class VGGStyleLoss2(torch.nn.Module):
    def __init__(self, transfer_mode=1, resize=True):
        super(VGGStyleLoss2, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True).to(device)
        for i, layer in enumerate(vgg.features):
            if isinstance(layer, torch.nn.MaxPool2d):
                vgg.features[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        blocks = []
        blocks.append(vgg.features[:4].eval())
        blocks.append(vgg.features[4:9].eval())
        blocks.append(vgg.features[9:16].eval())
        blocks.append(vgg.features[16:23].eval())
        blocks.append(vgg.features[23:].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)

        self.transform = torch.nn.functional.interpolate
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self.resize = resize

    def calc_mean_std(self,feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def forward(self, input, target):
        if target.max()>2:
            input=input/255.0
            target=target/255.0
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = torch.zeros(input.shape[0]).float().to(device)
        x = input
        y = target
        x_feats=[]
        y_feats=[]
        for block in self.blocks:
            x = block(x)
            y = block(y)
            x_feats.append(x)
            y_feats.append(y)
        for i in range(1, 5):
            s_feats_mean, s_feats_std = self.calc_mean_std(x_feats[i])
            stylized_feats_mean, stylized_feats_std = self.calc_mean_std(y_feats[i])
            loss +=((stylized_feats_mean-s_feats_mean)**2).mean(dim=(1,2,3)) + ((stylized_feats_std-s_feats_std)**2).mean(dim=(1,2,3))
        return loss
class MyStyleLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(MyStyleLoss, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True).to(device)
        for i, layer in enumerate(vgg.features):
            if isinstance(layer, torch.nn.MaxPool2d):
                vgg.features[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        blocks = []
        blocks.append(vgg.features[:4].eval())
        blocks.append(vgg.features[4:9].eval())
        blocks.append(vgg.features[9:16].eval())
        blocks.append(vgg.features[16:23].eval())
        print(blocks)
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)

        self.transform = torch.nn.functional.interpolate
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self.resize = resize

    def get_matrix(self, x,last_x):
        (b, ch, h, w) = last_x.size()
        x=F.interpolate(x,size=(h,w),mode='bilinear')
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * w * h)
        return gram

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = torch.zeros(input.shape[0]).float()
        x = input
        y = target
        last_x=self.blocks[0](x)
        last_y=self.blocks[0](y)
        for block in self.blocks[1:]:
            x = block(last_x)
            y = block(last_y)
            gm_x = self.get_matrix(x,last_x)
            gm_y = self.get_matrix(y,last_y)
            loss += ((gm_x-gm_y)**2).sum(dim=(1,2))
            last_x,last_y=x,y
        return loss
class StyleReward(torch.nn.Module):
    def __init__(self):
        super(StyleReward, self).__init__()
        self.style_loss = VGGStyleLoss(transfer_mode=1, resize=True)
        #self.content_loss=VGGPerceptualLoss()
        self.content_loss=PixelLoss()
        self.style_img=read_img('style.jpg').to(device)
    def forward(self,canvas,gt):
        loss_content = self.content_loss(canvas, gt)
        loss_style = self.style_loss(canvas, self.style_img)*0.1
        #loss_style = torch.zeros_like(loss_content).to(device).float()
        loss = loss_content + loss_style
        return loss,loss_content,loss_style
if __name__ == '__main__':
    net=VGGStyleLoss2().cuda()
    x1=torch.randn(5,3,128,128).float().cuda()
    x2=torch.randn(5,3,128,128).float().cuda()
    y=net(x1,x2)
    print(y.shape)
