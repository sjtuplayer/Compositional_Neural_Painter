import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn as nn
import morphology
import torch.nn.functional as F
def normal(x, width):
    return (int)(x * (width - 1) + 0.5)

def draw(f, width=128):
    x0, y0, x1, y1, x2, y2, z0, z2, w0, w2 = f
    x1 = x0 + (x2 - x0) * x1
    y1 = y0 + (y2 - y0) * y1
    x0 = normal(x0, width * 2)
    x1 = normal(x1, width * 2)
    x2 = normal(x2, width * 2)
    y0 = normal(y0, width * 2)
    y1 = normal(y1, width * 2)
    y2 = normal(y2, width * 2)
    z0 = (int)(1 + z0 * width // 2)
    z2 = (int)(1 + z2 * width // 2)
    canvas = np.zeros([width * 2, width * 2]).astype('float32')
    tmp = 1. / 100
    for i in range(100):
        t = i * tmp
        x = (int)((1-t) * (1-t) * x0 + 2 * t * (1-t) * x1 + t * t * x2)
        y = (int)((1-t) * (1-t) * y0 + 2 * t * (1-t) * y1 + t * t * y2)
        z = (int)((1-t) * z0 + t * z2)
        w = (1-t) * w0 + t * w2
        cv2.circle(canvas, (y, x), z, w, -1)
    return 1 - cv2.resize(canvas, dsize=(width, width))
def read_img(img_path, img_type='RGB'):
    img = Image.open(img_path).convert(img_type)
    img = np.array(img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float() / 255.
    return img
brush_large_vertical = read_img('brush/brush_large_vertical.png', 'L').cuda()
brush_large_horizontal = read_img('brush/brush_large_horizontal.png', 'L').cuda()
meta_brushes = torch.cat([brush_large_vertical, brush_large_horizontal], dim=0)
brush_large_vertical_pad = read_img('brush/brush_large_vertical_pad.png', 'L').cuda()
brush_large_horizontal_pad = read_img('brush/brush_large_horizontal_pad.png', 'L').cuda()
meta_brushes_pad = torch.cat([brush_large_vertical_pad, brush_large_horizontal_pad], dim=0)
def draw_oil(param, size=128):
    # param: b, 12
    H=W=size
    b = param.shape[0]
    param_list = torch.split(param, 1, dim=1)
    x0, y0, w, h, theta = [item.squeeze(-1) for item in param_list[:5]]
    sin_theta = torch.sin(torch.acos(torch.tensor(-1., device=param.device)) * theta)
    cos_theta = torch.cos(torch.acos(torch.tensor(-1., device=param.device)) * theta)
    index = torch.full((b,), -1, device=param.device)
    index[h > w] = 0
    index[h <= w] = 1
    brush = meta_brushes[index.long()]
    alphas = meta_brushes[index.long()]
    alphas = (alphas > 0).float()
    warp_00 = cos_theta / w
    warp_01 = sin_theta * H / (W * w)
    warp_02 = (1 - 2 * x0) * cos_theta / w + (1 - 2 * y0) * sin_theta * H / (W * w)
    warp_10 = -sin_theta * W / (H * h)
    warp_11 = cos_theta / h
    warp_12 = (1 - 2 * y0) * cos_theta / h - (1 - 2 * x0) * sin_theta * W / (H * h)
    warp_0 = torch.stack([warp_00, warp_01, warp_02], dim=1)
    warp_1 = torch.stack([warp_10, warp_11, warp_12], dim=1)
    warp = torch.stack([warp_0, warp_1], dim=1)
    grid = torch.nn.functional.affine_grid(warp, torch.Size((b, 3, H, W)), align_corners=False)
    brush = torch.nn.functional.grid_sample(brush, grid, align_corners=False)
    alphas = torch.nn.functional.grid_sample(alphas, grid, align_corners=False)
    brush=morphology.dilation(brush)
    #brush = morphology.erosion(brush)
    # print(brush[0].mean(),brush[-1].mean())
    # print(param[0,-1],param[-1,-1])
    #brush=brush*param[:,-1].view(-1,1,1)
    #print(brush[0].mean(), brush[-1].mean())
    #alphas = torch.nn.functional.grid_sample(alphas, grid, align_corners=False)
    return torch.cat([1-brush,1-alphas],dim=1)
    #return brush, alphas
def draw_oil_tps(param, tps,size=128):
    # param: b, 12
    def conv(x):
        x= F.conv2d(x, weight, padding=1)
        return x
    kernel = [[-1, -1, -1],
              [-1, 8, -1],
              [-1, -1, -1],
              ]
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0).cuda()
    weight = nn.Parameter(data=(kernel), requires_grad=True)
    H=W=size
    b = param.shape[0]
    param_list = torch.split(param, 1, dim=1)
    x0, y0, w, h, theta = [item.squeeze(-1) for item in param_list[:5]]
    controller_paras = [item for item in param_list[5:]]
    #print(torch.stack(controller_paras, dim=0).shape)
    controller_paras = torch.stack(controller_paras, dim=0).squeeze().transpose(1, 0).view(b, -1, 2)
    sin_theta = torch.sin(torch.acos(torch.tensor(-1., device=param.device)) * theta)
    cos_theta = torch.cos(torch.acos(torch.tensor(-1., device=param.device)) * theta)
    index = torch.full((b,), -1, device=param.device)
    index[h > w] = 0
    index[h <= w] = 1
    brush = meta_brushes_pad[index.long()]
    brush = tps.forward(brush, controller_paras)
    #alphas = meta_brushes_pad[index.long()]
    alphas = (brush > 0).float()
    warp_00 = cos_theta / w
    warp_01 = sin_theta * H / (W * w)
    warp_02 = (1 - 2 * x0) * cos_theta / w + (1 - 2 * y0) * sin_theta * H / (W * w)
    warp_10 = -sin_theta * W / (H * h)
    warp_11 = cos_theta / h
    warp_12 = (1 - 2 * y0) * cos_theta / h - (1 - 2 * x0) * sin_theta * W / (H * h)
    warp_0 = torch.stack([warp_00, warp_01, warp_02], dim=1)
    warp_1 = torch.stack([warp_10, warp_11, warp_12], dim=1)
    warp = torch.stack([warp_0, warp_1], dim=1)
    grid = torch.nn.functional.affine_grid(warp, torch.Size((b, 3, H, W)), align_corners=False)
    brush = torch.nn.functional.grid_sample(brush, grid, align_corners=False)
    alphas = torch.nn.functional.grid_sample(alphas, grid, align_corners=False)
    #brush = torch.clamp(morphology.dilation(brush), 0, 1)
    alphas = torch.clamp(morphology.erosion(alphas), 0, 1)
    edge = conv(alphas)
    edge=(edge>0.2).float()
    # print(brush[0].mean(),brush[-1].mean())
    # print(param[0,-1],param[-1,-1])
    #brush=brush*param[:,-1].view(-1,1,1)
    #print(brush[0].mean(), brush[-1].mean())
    #alphas = torch.nn.functional.grid_sample(alphas, grid, align_corners=False)
    return torch.cat([1-brush,1-alphas,1-edge],dim=1)
    #return brush, alphas
def draw_oil_edge(param, size=128):
    def conv(x):
        x= F.conv2d(x, weight, padding=1)
        return x
    kernel = [[-1, -1, -1],
              [-1, 8, -1],
              [-1, -1, -1],
              ]
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0).cuda()
    weight = nn.Parameter(data=(kernel), requires_grad=True)
    # param: b, 12
    H = W = size
    b = param.shape[0]
    param_list = torch.split(param, 1, dim=1)
    x0, y0, w, h, theta = [item.squeeze(-1) for item in param_list[:5]]
    sin_theta = torch.sin(torch.acos(torch.tensor(-1., device=param.device)) * theta)
    cos_theta = torch.cos(torch.acos(torch.tensor(-1., device=param.device)) * theta)
    index = torch.full((b,), -1, device=param.device)
    index[h > w] = 0
    index[h <= w] = 1
    brush = meta_brushes[index.long()]
    alphas = meta_brushes[index.long()]
    alphas = (alphas > 0).float()
    warp_00 = cos_theta / w
    warp_01 = sin_theta * H / (W * w)
    warp_02 = (1 - 2 * x0) * cos_theta / w + (1 - 2 * y0) * sin_theta * H / (W * w)
    warp_10 = -sin_theta * W / (H * h)
    warp_11 = cos_theta / h
    warp_12 = (1 - 2 * y0) * cos_theta / h - (1 - 2 * x0) * sin_theta * W / (H * h)
    warp_0 = torch.stack([warp_00, warp_01, warp_02], dim=1)
    warp_1 = torch.stack([warp_10, warp_11, warp_12], dim=1)
    warp = torch.stack([warp_0, warp_1], dim=1)
    grid = torch.nn.functional.affine_grid(warp, torch.Size((b, 3, H, W)), align_corners=False)
    brush = torch.nn.functional.grid_sample(brush, grid, align_corners=False)
    alphas = torch.nn.functional.grid_sample(alphas, grid, align_corners=False)
    # print(brush[0].mean(),brush[-1].mean())
    # print(param[0,-1],param[-1,-1])
    # brush=brush*param[:,-1].view(-1,1,1)
    # print(brush[0].mean(), brush[-1].mean())
    # alphas = torch.nn.functional.grid_sample(alphas, grid, align_corners=False)
    #brush = torch.clamp(morphology.dilation(brush), 0, 1)
    alphas=torch.clamp(morphology.erosion(alphas), 0, 1)
    edge = conv(alphas)
    edge = (edge > 0.2).float()
    #return torch.cat([1-alphas,1-torch.clamp(morphology.erosion(alphas), 0, 1),1-torch.clamp(morphology.erosion(morphology.erosion(alphas)), 0, 1)],dim=1)
    return torch.cat([1 - brush, 1 - alphas,1-edge], dim=1)
    # return brush, alphas
