import os
import cv2
import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.functional as F
from ipykernel.pickleutil import can
from style import losses
from Renderer.stroke_gen import *
from DRL.actor import *
from Renderer.stroke_gen import *
from torchvision.utils import save_image
from torchvision import transforms
from Renderer.network import FCN
import torch.optim as optim
from style import ada_loss,network
from PIL import Image
from style.DT import differentialble_distance_transform,edge_extractor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
width = 128

parser = argparse.ArgumentParser(description='Learning to Paint')
parser.add_argument('--max_step', default=30, type=int, help='max length for episode')
parser.add_argument('--actor', default='./baseline/model3/Paint-run1/actor.pkl', type=str, help='Actor model')
parser.add_argument('--renderer', default='./renderer-oil-FCN.pkl', type=str, help='renderer model')
parser.add_argument('--content_path', default='style/content_imgs/1.jpg', type=str, help='test image')
parser.add_argument('--style_path', default='style/style_imgs/wave.jpg', type=str, help='test image')
parser.add_argument('--output_path', default='output-style', type=str, help='test image')
parser.add_argument('--imgid', default=0, type=int, help='set begin number for generated image')
parser.add_argument('--d', default=4, type=int, help='d the target image to get better resolution')
parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
#parser.add_argument('--beta_style', default=1, type=float, help='learning rate')
parser.add_argument('--beta_content', default=1, type=float)
parser.add_argument('--beta_style', default=0, type=float)
parser.add_argument('--beta_dt', default=200, type=float, help='learning rate')
args = parser.parse_args()

origin_shape=(512,512)
param_num=5
Decoder = FCN(param_num,True,False)
Decoder.load_state_dict(torch.load(args.renderer))
def decode(x, canvas): # b * (10 + 3)
    x = x.view(-1, param_num + 3)
    foregrounds, alphas, _ = Decoder(x[:, :param_num + 3])
    foregrounds = foregrounds.view(-1, 5, 3, 128, 128)
    alphas = alphas.view(-1, 5, 1, 128, 128)
    res = []
    for i in range(5):
        canvas=canvas * (1 - alphas[:, i]) + alphas[:, i] * foregrounds[:, i]
        res.append(canvas)
    return canvas, res
def small2large(x,d=args.d):
    # (d * d, width, width) -> (d * width, d * width)
    x = x.reshape(d, d, width, width, -1)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(d * width, d * width, -1)
    return x

def large2small(x,d=args.d):
    # (d * width, d * width) -> (d * d, width, width)
    x = x.reshape(d, width, d, width, 3)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(canvas_cnt, width, width, 3)
    return x
def torch_large2small(x,d=args.d):
    # (d * width, d * width) -> (d * d, width, width)
    x=x.squeeze(0).permute(1,2,0)
    x = x.reshape(d, width, d, width,-1)
    x = torch.permute(x, (0, 2, 1, 3, 4))
    x = x.reshape(d**2,width, width,-1)
    x=x.permute((0,3,1,2))
    return x
def torch_small2large(x,d=args.d):
    # (d * d, width, width) -> (d * width, d * width)\
    x=x.permute(0,2,3,1)
    x = x.reshape(d, d, width, width,-1)
    x = torch.permute(x, (0, 2, 1, 3, 4))
    x = x.reshape(d * width, d * width,-1)
    x=x.permute((2,0,1))
    return x
def pad(img, H, W):
    c, h, w = img.shape
    pad_h = (H - h) // 2
    pad_w = (W - w) // 2
    remainder_h = (H - h) % 2
    remainder_w = (W - w) % 2
    img = torch.cat([torch.zeros((c, pad_h, w), device=img.device), img,
                     torch.zeros((c, pad_h + remainder_h, w), device=img.device)], dim=-2)
    img = torch.cat([torch.zeros((c, H, pad_w), device=img.device), img,
                     torch.zeros((c, H, pad_w + remainder_w), device=img.device)], dim=-1)
    return img
# param_num-=1
actor = ResNet(6, 18, 5*(param_num+3)) # action_bundle = 5, 65 = 5 * 13
actor.load_state_dict(torch.load(args.actor))
actor = actor.to(device).eval()
Decoder = Decoder.to(device).eval()

canvas = torch.zeros([1, 3, width, width]).to(device)
patch_img = cv2.resize(img, (width * args.d, width * args.d))
patch_img = large2small(patch_img)
patch_img = np.transpose(patch_img, (0, 3, 1, 2))
patch_img = torch.tensor(patch_img).to(device).float() / 255.
img_512=cv2.resize(img, (width*4, width*4))
img = cv2.resize(img, (width, width))
img = img.reshape(1, width, width, 3)
img = np.transpose(img, (0, 3, 1, 2))
img = torch.tensor(img).to(device).float() / 255.
os.system('mkdir output')
loss_mse=torch.nn.MSELoss()
resize_512=transforms.Resize((512,512))
old_params=[[],[],[]]
with torch.no_grad():
    for i in range(args.max_step):
        stepnum = args.max_step
        #actions = actor(torch.cat([canvas, img, stepnum, coord], 1))
        actions = actor(torch.cat([canvas, img], 1))
        old_params[0].append(actions.detach())
        canvas, res = decode(actions, canvas)
    if args.d != 1:
        canvas = canvas[0].detach().cpu().numpy()
        canvas = np.transpose(canvas, (1, 2, 0))
        canvas = cv2.resize(canvas, (width * args.d, width * args.d))
        canvas = large2small(canvas)
        canvas = np.transpose(canvas, (0, 3, 1, 2))
        canvas = torch.tensor(canvas).to(device).float()
        for i in range(args.max_step):
            stepnum = args.max_step
            actions = actor(torch.cat([canvas, patch_img], 1))
            old_params[1].append(actions)
            canvas, res = decode(actions, canvas)
        img = pad(resize_512(img).squeeze(0),640,640)
        canvas=torch_small2large(canvas)
        canvas=pad(canvas,640,640)
        canvas=torch_large2small(canvas,args.d+1)
        patch_img = torch_large2small(img, args.d + 1)
        for i in range(args.max_step):
            stepnum = args.max_step
            actions = actor(torch.cat([canvas, patch_img], 1))
            old_params[2].append(actions)
            canvas, res = decode(actions, canvas)
        canvas=torch_small2large(canvas,args.d+1)
params=[[],[],[]]
m=args.d
#x0, y0, w, h, theta
for param in old_params[0]:
    param=param.reshape(-1, 8)
    param[:,[5,6,7]]=param[:,[7,6,5]]
    params[0].append(param)
for param in old_params[1]:
    param = param.reshape(-1, 8)
    param[:, [5, 6, 7]] = param[:, [7, 6, 5]]
    params[1].append(param)
for param in old_params[2]:
    param = param.reshape(-1, 8)
    param[:, [5, 6, 7]] = param[:, [7, 6, 5]]
    params[2].append(param)
def param2img(params):
    resize = transforms.Resize((width*args.d,width*args.d))
    canvas = torch.zeros([1, 3, width, width]).to(device)
    for param in params[0]:
        canvas,res=decode(param, canvas)
    canvas=resize(canvas)
    canvas=torch_large2small(canvas[0])
    for param in params[1]:
        canvas, res = decode(param, canvas)
    canvas=torch_small2large(canvas)
    canvas=pad(canvas.squeeze(0),640,640)
    canvas=torch_large2small(canvas,args.d+1)
    for param in params[2]:
        canvas, res = decode(param, canvas)
    canvas=torch_small2large(canvas,args.d+1)
    return canvas[:,64:576,64:576]
canvas=param2img(params)
loader=transforms.Compose([
    transforms.Resize([512,512]),
    transforms.ToTensor()
])
content_img=loader(Image.open(args.content_path)).unsqueeze(0).cuda()
style_img=loader(Image.open(args.style_path)).unsqueeze(0).cuda()
for i in range(len(params[0])):
    params[0][i].requires_grad = True
for i in range(len(params[1])):
    params[1][i].requires_grad = True
for i in range(len(params[2])):
    params[2][i].requires_grad = True
optimizer = optim.Adam(params[0]+params[1]+params[2], lr=args.lr)
MSE_loss = torch.nn.MSELoss()
if args.style_mode==1:
    bs_content_layers = ['conv4_1', 'conv5_1']
    bs_style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    vgg_loss = losses.StyleTransferLosses('/home/huteng/brushstroke-parameterized-style-transfer-master/vgg_weights/vgg19_weights_normalized.h5', content_img, style_img,
                                        bs_content_layers, bs_style_layers, scale_by_y=True)
    vgg_loss.cuda().eval()
else:
    Ada_loss = ada_loss.AdaAttNModel(beta_content=args.beta_content)
if args.beta_dt!=0:
    edge_map=edge_extractor(content_img).cuda()
    print(edge_map.shape)
losses=dict()
tmp_lr=args.lr
l2_loss = torch.nn.MSELoss()
for epoch in range(1001):
    for i in range(len(params[0])):
        params[0][i].data = torch.clamp(params[0][i].data, 0,1)
    for i in range(len(params[1])):
        params[1][i].data = torch.clamp(params[1][i].data, 0,1)
    for i in range(len(params[2])):
        params[2][i].data = torch.clamp(params[2][i].data, 0,1)
    # if epoch%200==0 and epoch!=0:
    #     tmp_lr/=2
    canvas=param2img(params).unsqueeze(0)
    canvas=torch.clamp(canvas,0,1)
    loss = 0
    resize512=transforms.Resize([512,512])
    if epoch<100:
        loss=loss_mse(content_img,canvas)
    else:
        if args.style_mode==1:
            loss_content, loss_style = vgg_loss(canvas)
            loss = loss_content * args.beta_content + loss_style
        else:
            Ada_loss.set_input(resize512(content_img), resize512(style_img), resize512(canvas))
            loss=Ada_loss.compute_losses()
        if args.beta_dt != 0:
            edges = edges.mean(1).unsqueeze(0)
            edges = network.SignWithSigmoidGrad.apply(edges - 0.2)
            edges = 1 - edges
            dt = differentialble_distance_transform(edges, 30)
            loss_dt = (dt * edge_map).mean() * args.beta_dt
            loss += loss_dt
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(epoch,loss)
    if epoch % 10 == 0:
        print(os.path.join(args.output_path,'%d.jpg'%epoch))
        save_image(canvas[0], os.path.join(args.output_path,'%d.jpg'%epoch))

