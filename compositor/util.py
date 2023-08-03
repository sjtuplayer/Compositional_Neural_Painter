"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
# from thinplatespline.batch import TPS
# from Facecrop.FaceBoxes import FaceBoxes
# from landmark_detect.models.mobilefacenet import MobileFaceNet
from torchvision import transforms
import cv2
from thinplatespline.tps import tps_warp
from thinplatespline.utils import (
    TOTEN, TOPIL, grid_points_2d,  grid_to_img)
def drawLandmark(img,  landmark):
    img=img.permute((1,2,0)).detach().cpu().numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for index,(x, y) in enumerate(landmark):
        img = cv2.putText(img, str(index), (int(x), int(y)), font, 0.3, (255, 255, 255), 1)
        #cv2.circle(img, (int(x), int(y)), 3, (0,255,0), -1)
    return img
def torch_list_to_dict(x):
    y={}
    y['len']=len(x)
    for i in range(len(x)):
        y[str(i)]=x[i].cpu().detach().numpy()
    return y
def dict_to_torch_list(y,device):
    x=[]
    for i in range(y['len']):
        x.append(torch.from_numpy(y[str(i)]).to(device))
    return x

class Landmark_detector:
    def __init__(self):
        self.face_boxes = FaceBoxes(cuda=True)
        self.model = MobileFaceNet([112, 112], 136).cuda().eval()
        map_location = lambda storage, loc: storage.cuda()
        checkpoint = torch.load('landmark_detect/checkpoint/mobilefacenet_model_best.pth.tar',
                                map_location=map_location)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.resize=transforms.Resize((112,112))
    def detect(self,ori_img):
        if len(ori_img.shape)==4:
            ori_img=ori_img[0]
        det = self.face_boxes(ori_img[[2, 1, 0]])
        det = list(map(int, det))
        img = ori_img[:, det[1]:det[3], det[0]:det[2]]
        img=self.resize(img).unsqueeze(0)
        landmark = self.model(img)[0]
        landmark = landmark.reshape(-1, 2)
        h=det[3]-det[1]
        w=det[2]-det[0]
        ori_x,ori_y=det[0],det[1]
        new_landmark=torch.zeros_like(landmark).cuda()
        for i, point in enumerate(landmark):
            x = point[0] * w + ori_x
            y = point[1] * h + ori_y
            new_landmark[i][0]=x
            new_landmark[i][1]=y
        img = drawLandmark(ori_img*255.0, new_landmark)
        cv2.imwrite(os.path.join('output', 'test.png'), img)
        return new_landmark
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
def make_landmark_map(ori_landmarks,original_img):
    landmark_map=torch.zeros_like(original_img).float()
    sample_num=100
    points=[(0,16),(17,21),(22,26),(27,30),(31,35)]
    for p in points:
        for i in range(p[0],p[1]):
            x_s,y_s=ori_landmarks[i]
            x_e,y_e=ori_landmarks[i+1]
            delta_x=(x_e-x_s)/sample_num
            delta_y=(y_e-y_s)/sample_num
            x,y=x_s,y_s
            for j in range(sample_num+1):
                landmark_map[:,:,y.floor().int():y.ceil().int()+1,x.floor().int():x.ceil().int()+1]=1
                x=x+delta_x
                y=y+delta_y
    points = [(36,42),(42,48),(48,60)]
    for p in points:
        for i in range(p[0],p[1]):
            x_s,y_s=ori_landmarks[i]
            x_e,y_e=ori_landmarks[i+1] if i!=p[1]-1 else ori_landmarks[p[0]]
            delta_x=(x_e-x_s)/sample_num
            delta_y=(y_e-y_s)/sample_num
            x,y=x_s,y_s
            for j in range(sample_num+1):
                landmark_map[:,:,y.floor().int():y.ceil().int()+1,x.floor().int():x.ceil().int()+1]=1
                x=x+delta_x
                y=y+delta_y
    return landmark_map
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
