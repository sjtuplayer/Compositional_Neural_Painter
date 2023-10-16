import os
import cv2
import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.functional as F
from Renderer.stroke_gen import *
from DRL.actor import *
from Renderer import morphology
from PIL import Image
from torchvision.utils import save_image
from Renderer.network import FCN
from torchvision import transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
width = 128*4



parser = argparse.ArgumentParser(description='Learning to Paint')
parser.add_argument('--stroke_num', default=5000, type=int, help='max length for episode')
parser.add_argument('--compositor', default='compositor/checkpoints/compositor.pkl', type=str, help='Actor model')
parser.add_argument('--painter', default='painter/checkpoints/painter.pkl', type=str, help='Actor model')
parser.add_argument('--renderer', default='compositor/renderer-oil.pkl', type=str, help='renderer model')
parser.add_argument('--img_path', default='test-img/1.jpg', type=str, help='test image')
parser.add_argument('--mode', default=1, type=int, help='mode=1:compositor painting with size=512, mode=2:compositor painting with size=128, mode=3:5*5 blocks')
parser.add_argument('--video',  action='store_true', help='wheter to save_vedio')
args = parser.parse_args()

if args.video:
    fps = 10
    size=(512,512)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter('./video.avi', fourcc, fps, size)
    frame = cv2.imread('black.jpg')
    frame = cv2.resize(frame, size)
    videoWriter.write(frame)
    videoWriter.write(frame)

param_num=5
Decoder = FCN(param_num,True,True).to(device)
Decoder.load_state_dict(torch.load(args.renderer))
resize_128=transforms.Resize((128,128))
resize_64=transforms.Resize((64,64))
resize_512=transforms.Resize((512,512))
resize_256=transforms.Resize((256,256))
output_width=512
def oil_decoder(x,size=512):
    tmp = 1 - draw_oil(x[:, :param_num],size=size)
    stroke = tmp[:, 0]
    alpha = tmp[:, 1]
    stroke = stroke.view(-1, size, size, 1)
    alpha = alpha.view(-1, size, size, 1)
    color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
    alpha = alpha.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)
    alpha = alpha.view(-1, 5, 1, size, size)
    color_stroke = color_stroke.view(-1, 5, 3, size, size)
    return  color_stroke,alpha
def decode(box, canvas, tar_canvas,debug=False):  # b * (10 + 3)
    ori_canvas=canvas.clone()
    canvas=resize_128(canvas)
    tar_canvas=resize_128(tar_canvas)
    for i in range(canvas.size(0)):
        x1, y1, x2, y2 = torch.round(box[i]*127).detach().int()
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        resize = transforms.Resize((4 * (x2 + 1 - x1), 4 * (y2 + 1 - y1)))
        tar_canvas_box=resize_128(tar_canvas[i, :, x1:x2 + 1, y1:y2 + 1]).unsqueeze(0)
        for kk in range(1):
            canvas = resize_128(ori_canvas)
            canvas_box = resize_128(canvas[i, :, x1:x2 + 1, y1:y2 + 1]).unsqueeze(0)
            param = painter(torch.cat((canvas_box, tar_canvas_box), dim=1))
            params.append(param)
            x = param.view(-1, param_num + 3)
            # foregrounds, alphas, _ = Decoder(x[:, :param_num + 3])
            foregrounds, alphas = oil_decoder(x[:, :param_num + 3])
            foregrounds = foregrounds.view(-1, 5, 3, 512, 512)
            alphas = alphas.view(-1, 5, 1, 512, 512)
            for j in range(5):
                ori_canvas[i, :, 4*x1:4*(x2+1), 4*y1:4*y2+4] = \
                    ori_canvas[i, :, 4*x1:4*(x2+1), 4*y1:4*y2+4] * resize(1 - alphas[0, j]) \
                    + resize(alphas[0, j]) * resize(foregrounds[0, j])
    return ori_canvas
def final_decode():
    cnt=0
    canvas=torch.zeros(1,3,output_width,output_width).cuda()
    for index,box0 in enumerate(boxes0):
        x01,y01,x02,y02=box0
        x01,y01,x02,y02=min(x01,x02),min(y01,y02),max(x01,x02),max(y01,y02)
        # x01, y01, x02, y02=(x01*512).int(), (y01).int(), (x02).int(), (y02).int()
        w0=x02-x01
        h0=y02-y01
        for index1 in range(recursive_number):
            x11, y11, x12, y12 = boxes1.pop(0)
            x11, y11, x12, y12 = min(x11, x12), min(y11, y12), max(x11, x12), max(y11, y12)
            x1=int((x01+x11*w0)*(output_width-1))
            x2=int((x01+x12*w0)*(output_width-1))
            y1=int((y01+y11*h0)*(output_width-1))
            y2=int((y01+y12*h0)*(output_width-1))
            resize=transforms.Resize((x2+1-x1,y2+1-y1))
            for k in range(1):
                param=params.pop(0)
                x = param.view(-1, param_num + 3)
                foregrounds, alphas = oil_decoder(x[:, :param_num + 3])
                foregrounds = foregrounds.view(-1, 5, 3, 512, 512)
                alphas = alphas.view(-1, 5, 1, 512, 512)
                # foregrounds[0] = morphology.dilation(foregrounds[0])
                # alphas[0]=morphology.erosion(alphas[0])
                for j in range(5):
                    canvas[0, :, x1: (x2 + 1),  y1: y2 + 1] = \
                        canvas[0, :, x1: (x2 + 1),  y1: y2 + 1] * resize(1 - alphas[0, j]) \
                        + resize(alphas[0, j]) * resize(foregrounds[0, j])
                cnt+=1
                if args.video:
                    if cnt < 60:
                        frame = (canvas[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        videoWriter.write(frame)
                    elif cnt % 20 == 0:
                        frame = (canvas[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        videoWriter.write(frame)
    return canvas

actor = ResNet(6, 18, 4) # canvas,target
actor.load_state_dict(torch.load(args.compositor))
actor = actor.to(device).eval()
painter=ResNet(6, 18, 5*(param_num+3))
painter.load_state_dict(torch.load(args.painter))
painter = painter.to(device).eval()

canvas = torch.zeros([1, 3, width, width]).to(device)

loader = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize([512, 512])
            ])
losses=0
loss_mse = torch.nn.MSELoss()

if args.mode!=3:
    steps = args.stroke_num // 5
    recursive_number = 1
    if steps > 200:
        recursive_number = steps // 200 + 1
        steps = steps // recursive_number + 1
    boxes0 = []
    boxes1 = []
    params = []
    img = Image.open(args.img_path).convert('RGB')
    image = loader(img).unsqueeze(0).cuda()
    if args.mode==1:
        image=resize_512(image)
    if args.mode==2:
        image=resize_512(resize_128(image))
    image=image[:,[2,1,0]]
    with torch.no_grad():
        for i in range(steps):
            box = actor(torch.cat([canvas, image], 1))
            boxes0.append(box[0].detach())
            x1, y1, x2, y2 = torch.round(box[0] * 511).detach().int()
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            resize = transforms.Resize(((x2 + 1 - x1), (y2 + 1 - y1)))
            tar_canvas_box = resize_512(image[0, :, x1:x2 + 1, y1:y2 + 1]).unsqueeze(0)
            tmp_canvas_box = resize_512(canvas[0, :, x1:x2 + 1, y1:y2 + 1]).unsqueeze(0)
            for j in range(recursive_number):
                actions = actor(torch.cat([tmp_canvas_box, tar_canvas_box], 1))
                boxes1.append(actions[0].detach())
                tmp_canvas_box = decode(actions, tmp_canvas_box,tar_canvas_box)
            canvas[0,:,x1:x2 + 1, y1:y2 + 1]=resize(tmp_canvas_box[0])
        pixel_loss = loss_mse(canvas,image )
        losses+=float(pixel_loss.detach())
        canvas=final_decode()
        pixel_loss = loss_mse(canvas, image)
        save_image(canvas[:, [2, 1, 0]], 'output.png', nrow=1, normalize=False)
    print('MSE Distance',losses)
    if args.video:
        videoWriter.release()
else:
    K=5
    canvas_cnt=K*K
    width=128
    def small2large(x):
        # (d * d, width, width) -> (d * width, d * width)
        x = x.reshape(K, K, width, width, -1)
        x = np.transpose(x, (0, 2, 1, 3, 4))
        x = x.reshape(K * width, K * width, -1)
        return x
    def large2small(x):
        # (d * width, d * width) -> (d * d, width, width)
        x = x.reshape(K, width, K, width, 3)
        x = np.transpose(x, (0, 2, 1, 3, 4))
        x = x.reshape(canvas_cnt, width, width, 3)
        return x


    def decode3(x, canvas):  # b * (10 + 3)
        x = x.view(-1, param_num + 3)
        tmp = 1 - draw_oil(x[:, :param_num])
        stroke = tmp[:, 0]
        alpha = tmp[:, 1]
        stroke = stroke.view(-1, 128, 128, 1)
        alpha = alpha.view(-1, 128, 128, 1)
        color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
        alpha = alpha.permute(0, 3, 1, 2)
        color_stroke = color_stroke.permute(0, 3, 1, 2)
        alpha = alpha.view(-1, 5, 1, 128, 128)
        color_stroke = color_stroke.view(-1, 5, 3, 128, 128)
        res = []
        for i in range(5):
            canvas = canvas * (1 - alpha[:, i]) + alpha[:, i] * color_stroke[:, i]
            res.append(canvas)
        return canvas, res

    img = cv2.imread(args.img_path, cv2.IMREAD_COLOR)
    origin_shape = (512, 512)
    canvas = torch.zeros([1, 3, width, width]).to(device)
    patch_img = cv2.resize(img, (width * K, width * K))
    patch_img = large2small(patch_img)
    patch_img = np.transpose(patch_img, (0, 3, 1, 2))
    patch_img = torch.tensor(patch_img).to(device).float() / 255.
    img = cv2.resize(img, (width, width))
    img = img.reshape(1, width, width, 3)
    img = np.transpose(img, (0, 3, 1, 2))
    img = torch.tensor(img).to(device).float() / 255.
    steps=args.stroke_num//(canvas_cnt+1)
    with torch.no_grad():
        for i in range(steps):
            actions = painter(torch.cat([canvas, img], 1))
            canvas, res = decode3(actions, canvas)
        canvas = canvas[0].detach().cpu().numpy()
        canvas = np.transpose(canvas, (1, 2, 0))
        canvas = cv2.resize(canvas, (width * K, width * K))
        canvas = large2small(canvas)
        canvas = np.transpose(canvas, (0, 3, 1, 2))
        canvas = torch.tensor(canvas).to(device).float()
        for i in range(steps):
            actions = painter(torch.cat([canvas, patch_img], 1))
            canvas, res = decode3(actions, canvas)
        pixel_loss = loss_mse(canvas, patch_img)
        print('MSE Distance',pixel_loss)
        output = res[-1].detach().cpu().numpy()  # d * d, 3, width, width
        output = np.transpose(output, (0, 2, 3, 1))
        output = small2large(output)
        output = (output * 255).astype('uint8')
        output = cv2.resize(output, origin_shape)
        cv2.imwrite('output.png', output)

