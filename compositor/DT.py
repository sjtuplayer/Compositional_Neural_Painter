import math
import cv2
import torch
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import argparse
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
import torch.nn as nn
from scipy import ndimage
class SignWithSigmoidGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        result = (x > 0).float()
        sigmoid_result = torch.sigmoid(x)
        ctx.save_for_backward(sigmoid_result)
        return result

    @staticmethod
    def backward(ctx, grad_result):
        (sigmoid_result,) = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = grad_result * sigmoid_result * (1 - sigmoid_result)
        else:
            grad_input = None
        return grad_input
def read_img(img_path, img_type='RGB', h=None, w=None):
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
def show_tensor_img(img):
    plt.imshow(img[0].cpu().detach().numpy().transpose((1, 2, 0)))
    #plt.show()
    plt.pause(0.01)
def test():
    x=read_img('edge.jpg','L',h=256,w=256).cuda()
    x=(x>0.1).int()
    x=SignWithSigmoidGrad.apply(x-0.2)
    x=1-x
    y = torch.from_numpy(ndimage.distance_transform_edt(x.cpu().numpy())).float().cuda()
    print(y.max(),y.min())
    y_hat = differentialble_distance_transform(x,30)
    z = torch.cat((y,  y_hat), dim=3)
    #save_image(y,'1.jpg',normalize=False)
    show_tensor_img(y_hat)
def differentialble_distance_transform(image, kernel_size,c=0.3):
    # image=image.squeeze(0)
    image=1-image
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    half_size = kernel_size // 2
    unfold_nopad = nn.Unfold(kernel_size=kernel_size, padding=0, stride=1)
    unfold_pad = nn.Unfold(kernel_size=kernel_size,  padding=half_size, stride=1)
    fold = torch.nn.Fold(output_size=(image.size(-2), image.size(-1)), kernel_size=1, stride=1)
    kernel=torch.zeros(kernel_size,kernel_size).cuda().float()
    for i in range(-half_size,half_size+1):
        for j in range(-half_size,half_size+1):
            kernel[half_size+i][half_size+j]=torch.sqrt(torch.tensor(i**2+j**2))
    max_d=kernel[0][0]
    kernel=kernel.unsqueeze(0).unsqueeze(0)
    unfold_kernel = unfold_nopad(kernel)
    if kernel_size % 2 == 1:
        kernel_size = kernel_size + 1
    unfold_image=unfold_pad(image)
    dt_image=unfold_image*unfold_kernel+(1-unfold_image)*max_d
    dt_image=(dt_image*torch.exp(-dt_image/c)/(torch.exp(-dt_image/c).sum(1))).sum(1).unsqueeze(1)
    dt_image=fold(dt_image)
    return dt_image
def edge_extractor(img0,blur=11):
    img=(img0*255).cpu().squeeze(0).numpy().astype(np.uint8).transpose(1,2,0)
    blurred = cv2.GaussianBlur(img, (blur, blur), 0)
    gaussImg = cv2.Canny(blurred, 10, 70)
    return torch.from_numpy(gaussImg).unsqueeze(0).unsqueeze(0)/255.0
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='DT')
    parser.add_argument('--img_path', type=str,default='image/41.jpg')
    args = parser.parse_args()
    x = read_img(args.img_path, 'L', h=256, w=256)
    x=edge_extractor(x,7).cuda()
    x = (x > 0.1).int()
    x = SignWithSigmoidGrad.apply(x - 0.2)
    x=1-x
    dt_map=differentialble_distance_transform(x, 30)
    print(dt_map.max(),dt_map.min())
    show_tensor_img(dt_map)