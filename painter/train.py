from torchvision import transforms
from torch.utils.data import Dataset
import torch
import os
import argparse
from PIL import Image
from torch.utils.data import DataLoader
from models import networks
from torch import nn
from torchvision.utils import save_image
from util import morphology,util
from scipy.optimize import linear_sum_assignment
from torchvision import transforms
import numpy as np
from buffer import memory
from models import critic
from models import wgan
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
class Coach():
    def __init__(self,args):
        self.opts=args
        opts=args
        self.d_shape=5
        self.d=self.d_shape+3
        self.used_strokes=5
        self.size=opts.size
        self.blur_size = 16
        self.blur_resize=transforms.Resize((self.blur_size,self.blur_size))
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resize=transforms.Resize([self.size,self.size])
        # self.Paint_Transformer = networks.Painter(5, 8, 256,n_enc_layers=3, n_dec_layers=3)
        # self.Paint_Transformer.load_state_dict(torch.load('model.pth'))
        self.net_g = networks.ResNet(6, 18, self.used_strokes * (self.d)).to(self.device)
        self.critic=wgan.Wgan(self.opts,input_dim=args.w_dim,dataset_inside=False)
        if args.resume_iter!=-1:
            self.net_g.load_state_dict(torch.load(os.path.join(args.save_dir,'E-%s-%d.pth'%(args.dataset,args.resume_iter))))
            self.critic.load(os.path.join(args.save_dir,'Critic-%s-%d.pth'%(args.dataset,args.resume_iter)))
        self.net_g.train()
        self.G = networks.FCN(5,True,False).to(self.device)
        self.G.load_state_dict(torch.load('../oil_brush.pkl'))
        self.optimizer = torch.optim.Adam(self.net_g.parameters(), lr=opts.lr, betas=(opts.beta1, 0.999))
        self.global_step = opts.load_iter
        self.loss_mse = nn.MSELoss()
        self.loss_l1=nn.L1Loss()
        self.buffer=memory()
        self.configure_datasets()

    def sample(self,target):
        with torch.no_grad():
            losses={}
            target = target.to(self.device)
            canvas = torch.zeros(self.opts.batch_size, 3, self.opts.size, self.opts.size).cuda()
            for ii in range(self.opts.train_iter):
                param = self.net_g(target, canvas)
                param = param.view(-1, self.d).contiguous()
                foregrounds, alphas = self.param2stroke_G(param)
                foregrounds = torch.clamp(foregrounds.view(-1, self.used_strokes, 3, self.size, self.size), 0, 1)
                alphas = torch.clamp(alphas.view(-1, self.used_strokes, 3, self.size, self.size), 0, 1)
                for j in range(foregrounds.shape[1]):
                    foreground = foregrounds[:, j, :, :, :]
                    alpha = alphas[:, j, :, :, :]
                    canvas = foreground * alpha + canvas * (1 - alpha)
                self.buffer.append(torch.cat((canvas,target),dim=1),ii)
            if self.opts.blur and self.global_step%200==0 and self.global_step!=0:
                self.blur_size+=1
                print('update_blur_size:%d'%self.blur_size)
                self.blur_resize = transforms.Resize((self.blur_size, self.blur_size))
            if self.global_step % 10 == 0:
                pixel_loss = self.loss_mse(target, canvas)
                losses['pixel_loss'] = round(float(pixel_loss.cpu().detach()), 4)
                if self.opts.blur:
                    blur_img=self.blur(target)
                    pixel_blur_loss = self.loss_mse(blur_img, canvas)
                    losses['pixel_blur_loss'] = round(float(pixel_blur_loss.cpu().detach()), 4)
                with open(os.path.join(args.save_dir, 'loss.txt'), 'a')as f:
                    f.write('step:%d,pixel_loss(mse):%f\n' % (self.global_step, pixel_loss))
                print(self.global_step,losses)
                save_img = []
                save_img.append(target[:3])
                save_img.append(canvas[:3])
                if self.opts.blur:
                    save_img.append(blur_img[:3])
                else:
                    save_img.append(foreground[:3])
                save_img = torch.cat(save_img, dim=0)
                save_image(save_img,
                           os.path.join(self.opts.save_dir, "images/%d.png") % (self.global_step),
                           nrow=3, normalize=False)
    def blur(self, img):
        return self.resize(self.blur_resize(img))
    def train(self):
        if self.opts.resume_iter==-1:
            with open(os.path.join(self.opts.save_dir, 'loss.txt'), 'w')as f:
                pass
        self.global_step=self.opts.resume_iter+1
        self.global_step=0
        while self.global_step < 2000000:
            self.losses={}
            for batch_idx, batch in enumerate(self.train_dataloader):
                self.sample(batch)
                if batch_idx<self.opts.warm_up:
                    continue
                self.buffer.random_reset()
                for i in range(int(self.opts.train_iter)):
                    state,step=self.buffer.pop()
                    # self.param_loss(state, step)
                    canvas = state[:, :3].to(self.device)
                    target_canvas = state[:, 3:].to(self.device)
                    param = self.net_g(target_canvas, canvas)
                    param = param.view(-1, self.d).contiguous()
                    foregrounds, alphas = self.param2stroke_G(param)
                    foregrounds = torch.clamp(foregrounds.view(-1, self.used_strokes, 3, self.size, self.size), 0, 1)
                    alphas = torch.clamp(alphas.view(-1, self.used_strokes, 3, self.size, self.size), 0, 1)
                    for j in range(foregrounds.shape[1]):
                        foreground = foregrounds[:, j, :, :, :]
                        alpha = alphas[:, j, :, :, :]
                        canvas = foreground * alpha + canvas * (1 - alpha)
                    loss=0
                    if self.opts.beta_w!=0 and self.opts.beta_pixel!=0:
                        true_canvas = torch.rand_like(target_canvas).cuda()
                        pixel_loss = self.loss_mse(canvas, target_canvas) * self.opts.beta_pixel
                        loss += pixel_loss
                        D_fake, D_real, gradient_penalty = self.critic.update(canvas, true_canvas)
                        gan_loss = -self.critic.cal_reward(canvas, true_canvas).mean()
                        gan_loss = gan_loss/float(gan_loss.detach())*float(pixel_loss.detach())*self.opts.beta_w
                        loss += gan_loss
                    elif self.opts.beta_pixel!=0:
                        pixel_loss=self.loss_mse(canvas,target_canvas)*self.opts.beta_pixel
                        loss += pixel_loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                if self.global_step % 10 == 0:
                    if self.opts.beta_w!=0:
                        self.losses['gan_loss'] = round(float(gan_loss.detach()), 4)
                        self.losses['D_real']=round(float(D_real.detach()), 4)
                        self.losses['D_fake'] = round(float(D_fake.detach()), 4)
                    if self.opts.beta_pixel != 0:
                        self.losses['pixel_loss'] = round(float(pixel_loss.detach()), 4)
                    self.losses['total_loss'] = round(float(loss.detach()), 4)
                    print(self.global_step, self.losses)
                if self.global_step%self.opts.save_model_iter:
                    torch.save(self.net_g.state_dict(),self.opts.save_dir + "/Painter.pth")
                self.global_step += 1
    def param2stroke_G(self, param):
        valid_foregrounds, valid_alphas = self.G(param.unsqueeze(-1).unsqueeze(-1))
        valid_foregrounds = morphology.dilation(valid_foregrounds)
        valid_alphas = morphology.erosion(valid_alphas)
        valid_alphas=self.resize(valid_alphas)
        valid_foregrounds=self.resize(valid_foregrounds)
        if valid_alphas.size(1)==1:
            valid_alphas=valid_alphas.repeat(1,3,1,1)
        return valid_foregrounds,valid_alphas
    def configure_datasets(self):

        class Data(Dataset):
            def __init__(self, img_path):
                self.loader = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize([128, 128])
                ])
                self.data_path = img_path
                self.file_names = os.listdir(self.data_path)
                self.l = len(self.file_names)
                print(self.l)

            def __getitem__(self, idx):
                image = Image.open(os.path.join(self.data_path, self.file_names[idx])).convert('RGB')
                image = self.loader(image)
                return image

            def __len__(self):
                return self.l
        self.train_data = Data(self.opts.train_dataset)
        self.test_data = Data(self.opts.test_dataset)
        self.train_dataloader = DataLoader(self.train_data,
                                           batch_size=self.opts.batch_size,
                                           shuffle=True,
                                           num_workers=8,
                                           drop_last=True)
        self.test_dataloader = DataLoader(self.test_data,
                                          batch_size=self.opts.batch_size,
                                          shuffle=True,
                                          num_workers=8,
                                          drop_last=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--train_dataset', type=str)
    parser.add_argument('--test_dataset', type=str)
    parser.add_argument('--display', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--print_iter', type=int, default=50)
    parser.add_argument('--save_model_iter', type=int, default=500)
    parser.add_argument('--save_img_iter', type=int, default=50)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--dataset', type=str, default='celeba')
    parser.add_argument('--loss_file', type=str, default='loss.txt')
    parser.add_argument('--load_iter', type=int, default=0)
    parser.add_argument('--layer_num', type=int, default=3)
    parser.add_argument('--resume_iter', type=int, default=-1)
    parser.add_argument('--warm_up', type=int, default=20)
    parser.add_argument('--train_iter', type=int, default=40)
    parser.add_argument('--beta_lpips', type=float, default=0)
    parser.add_argument('--beta_step', type=float, default=0)
    parser.add_argument('--beta_w', type=float, default=0.1)
    parser.add_argument('--beta_pixel', type=float, default=1)
    parser.add_argument('--w_dim', type=int, default=3)
    parser.add_argument('--blur', action='store_true', default=False)
    args = parser.parse_args()
    coach=Coach(args)
    coach.train()