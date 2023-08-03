"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
#from thinplatespline.batch import TPS
import torch.nn as nn
import util.pytorch_batch_sinkhorn as spc
import random
#from thinplatespline.tps import tps_warp
# from thinplatespline.utils import (
#     TOTEN, TOPIL, grid_points_2d,  grid_to_img)
def gaussian_w_distance(param_1, param_2):
    def get_sigma_sqrt(w, h, theta):
        sigma_00 = w * (torch.cos(theta) ** 2) / 2 + h * (torch.sin(theta) ** 2) / 2
        sigma_01 = (w - h) * torch.cos(theta) * torch.sin(theta) / 2
        sigma_11 = h * (torch.cos(theta) ** 2) / 2 + w * (torch.sin(theta) ** 2) / 2
        sigma_0 = torch.stack([sigma_00, sigma_01], dim=-1)
        sigma_1 = torch.stack([sigma_01, sigma_11], dim=-1)
        sigma = torch.stack([sigma_0, sigma_1], dim=-2)
        return sigma

    def get_sigma(w, h, theta):
        sigma_00 = w * w * (torch.cos(theta) ** 2) / 4 + h * h * (torch.sin(theta) ** 2) / 4
        sigma_01 = (w * w - h * h) * torch.cos(theta) * torch.sin(theta) / 4
        sigma_11 = h * h * (torch.cos(theta) ** 2) / 4 + w * w * (torch.sin(theta) ** 2) / 4
        sigma_0 = torch.stack([sigma_00, sigma_01], dim=-1)
        sigma_1 = torch.stack([sigma_01, sigma_11], dim=-1)
        sigma = torch.stack([sigma_0, sigma_1], dim=-2)
        return sigma
    mu_1, w_1, h_1, theta_1 = torch.split(param_1, (2, 1, 1, 1), dim=-1)
    w_1 = w_1.squeeze(-1)
    h_1 = h_1.squeeze(-1)
    theta_1 = torch.acos(torch.tensor(-1., device=param_1.device)) * theta_1.squeeze(-1)
    trace_1 = (w_1 ** 2 + h_1 ** 2) / 4
    mu_2, w_2, h_2, theta_2 = torch.split(param_2, (2, 1, 1, 1), dim=-1)
    w_2 = w_2.squeeze(-1)
    h_2 = h_2.squeeze(-1)
    theta_2 = torch.acos(torch.tensor(-1., device=param_2.device)) * theta_2.squeeze(-1)
    trace_2 = (w_2 ** 2 + h_2 ** 2) / 4
    sigma_1_sqrt = get_sigma_sqrt(w_1, h_1, theta_1)
    sigma_2 = get_sigma(w_2, h_2, theta_2)
    trace_12 = torch.matmul(torch.matmul(sigma_1_sqrt, sigma_2), sigma_1_sqrt)
    trace_12 = torch.sqrt(trace_12[..., 0, 0] + trace_12[..., 1, 1] + 2 * torch.sqrt(
        trace_12[..., 0, 0] * trace_12[..., 1, 1] - trace_12[..., 0, 1] * trace_12[..., 1, 0]))
    return torch.sum((mu_1 - mu_2) ** 2, dim=-1) + trace_1 + trace_2 - 2 * trace_12
class tps:
    def __init__(self,w,h,device):
        self.w=w
        self.h=h
        self.device=device
        self.X=self.initialize(w,h,device)
    def initialize(self,width,height,device):
        xx = torch.linspace(-0.5, 0.5, height, device=device)
        yy = torch.ones_like(xx) * 0.5
        tmp = torch.linspace(-0.5, 0.5, height, device=device)
        xx = torch.cat((xx, tmp), dim=0)
        yy = torch.cat((yy, -torch.ones_like(tmp) * 0.5), dim=0)
        tmp = torch.linspace(-0.5, 0.5, width, device=device)[1:-1]
        yy = torch.cat((yy, tmp), dim=0)
        xx = torch.cat((xx, -torch.ones_like(tmp) * 0.5), dim=0)
        tmp = torch.linspace(-0.5, 0.5, width, device=device)[1:-1]
        yy = torch.cat((yy, tmp), dim=0)
        xx = torch.cat((xx, torch.ones_like(tmp) * 0.5), dim=0)
        return torch.stack([yy, xx], dim=-1).contiguous().view(-1, 2).unsqueeze(0)
    def forward(self,img,Y):
        w,h=img.shape[-2:]
        X = self.X.repeat(Y.size(0),1,1).contiguous()
        Y=X+Y
        Y=torch.clamp(Y,-1,1)
        tpsb = TPS(size=(h, w), device=self.device)
        warped_grid_b = tpsb(X, Y)
        ten_wrp_b = torch.grid_sampler_2d(
            img,
            warped_grid_b,
            0, 0, False)
        return ten_wrp_b

class SinkhornLoss(nn.Module):

    def __init__(self, epsilon=0.01, niter=5, normalize=False,device='cuda'):
        super(SinkhornLoss, self).__init__()
        self.epsilon = epsilon
        self.niter = niter
        self.normalize = normalize
        self.device=device
    def _mesh_grids(self, batch_size, h, w):

        a = torch.linspace(0.0, h - 1.0, h).to(self.device)
        b = torch.linspace(0.0, w - 1.0, w).to(self.device)
        y_grid = a.view(-1, 1).repeat(batch_size, 1, w) / h
        x_grid = b.view(1, -1).repeat(batch_size, h, 1) / w
        grids = torch.cat([y_grid.view(batch_size, -1, 1), x_grid.view(batch_size, -1, 1)], dim=-1)
        return grids

    def forward(self, canvas, gt):

        batch_size, c, h, w = gt.shape
        if h > 24:
            canvas = nn.functional.interpolate(canvas, [48, 48], mode='area')
            gt = nn.functional.interpolate(gt, [48, 48], mode='area')
            batch_size, c, h, w = gt.shape

        canvas_grids = self._mesh_grids(batch_size, h, w)
        gt_grids = torch.clone(canvas_grids)

        # randomly select a color channel, to speedup and consume memory
        i = random.randint(0, 2)

        img_1 = canvas[:, [i], :, :]
        img_2 = gt[:, [i], :, :]

        mass_x = img_1.reshape(batch_size, -1)
        mass_y = img_2.reshape(batch_size, -1)
        if self.normalize:
            loss = spc.sinkhorn_normalized(
                canvas_grids, gt_grids, epsilon=self.epsilon, niter=self.niter,
                mass_x=mass_x, mass_y=mass_y)
        else:
            loss = spc.sinkhorn_loss(
                canvas_grids, gt_grids, epsilon=self.epsilon, niter=self.niter,
                mass_x=mass_x, mass_y=mass_y)


        return loss

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0  # post-processing: transpose and scaling
    else:  # if it is a numpy array
        image_numpy = input_image * 255.
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
