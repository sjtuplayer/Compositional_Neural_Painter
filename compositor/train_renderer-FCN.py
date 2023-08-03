import cv2
import torch
import numpy as np
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from utils.tensorboard import TensorBoard
from Renderer.model import FCN
from Renderer.stroke_gen import *
writer = TensorBoard("./train_log/")
import torch.optim as optim
from torchvision.utils import save_image
import util
from Renderer.network import UnetGenerator
criterion = nn.MSELoss()
stroke_num=5
net = FCN(stroke_num,True,False)
optimizer = optim.Adam(net.parameters(), lr=3e-6)
batch_size = 32
use_cuda = torch.cuda.is_available()
step = 0

def save_model():
    if use_cuda:
        net.cpu()
    torch.save(net.state_dict(), "./oil_brush.pkl")
    if use_cuda:
        net.cuda()


def load_weights():
    pretrained_dict = torch.load("./oil_brush.pkl")
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)


# load_weights()
step=0
while step < 500000+5:
    net.train()
    train_batch=torch.rand(batch_size,stroke_num).cuda()
    train_batch[:, 2:4] = train_batch[:, 2:4] * 2
    train_batch[:, 5:] = (train_batch[:, 5:] - 0.5) * 0.4
    with torch.no_grad():
        ground_truth=draw_oil(train_batch)
    if use_cuda:
        net = net.cuda()
        train_batch = train_batch.cuda()
        ground_truth = ground_truth.cuda()
    gen = net(train_batch)
    optimizer.zero_grad()
    loss = criterion(gen[:,:2], ground_truth[:,:2])
    loss.backward()
    optimizer.step()
    if step < 200000:
        lr = 1e-6
    elif step < 400000:
        lr = 1e-6
    else:
        lr = 1e-6
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    writer.add_scalar("train/loss", loss.item(), step)
    if step % 1000 == 0:
        print(step, loss.item())
        net.eval()
        gen = net(train_batch)
        loss = criterion(gen[:,:2], ground_truth[:,:2])
        print(float(loss))
        writer.add_scalar("val/loss", loss.item(), step)
        save_imgs = torch.cat([ground_truth[:1, 0], gen[:1, 0],
                               ground_truth[:1, 1], gen[:1, 1]],dim=0).unsqueeze(1)

        save_image(save_imgs,'./output/%d.jpg'%step,nrow=2)
    if step % 10000 == 0 and step!=0:
        save_model()
    step += 1
