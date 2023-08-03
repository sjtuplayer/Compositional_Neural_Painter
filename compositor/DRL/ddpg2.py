import os.path
import random
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from DRL.rpm import rpm
from DRL.actor import *
from DRL.critic import *
from DRL.wgan import *
from utils.util import *
from DRL.loss import *
from Renderer.network import *
from torchvision import transforms
from torchvision.utils import save_image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
width=128
coord = torch.zeros([1, 2, 128, 128])
for i in range(128):
    for j in range(128):
        coord[0, 0, i, j] = i / 127.
        coord[0, 1, i, j] = j / 127.
coord = coord.to(device)

criterion = nn.MSELoss()
param_num=5
Decoder = FCN(param_num,True,True)
Decoder.load_state_dict(torch.load('../renderer-oil-FCN.pkl'))
# Decoder=nn.DataParallel(Decoder).cuda()
# def decode(x, canvas): # b * (10 + 3)透明笔触
#     #print(x.shape)
#     x = x.view(-1, param_num + 3)
#     stroke = 1 - Decoder(x[:, :param_num])
#     #print(stroke.shape)
#     stroke = stroke.view(-1, 128, 128, 1)
#     color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
#     stroke = stroke.permute(0, 3, 1, 2)
#     color_stroke = color_stroke.permute(0, 3, 1, 2)
#     stroke = stroke.view(-1, 5, 1, 128, 128)
#     color_stroke = color_stroke.view(-1, 5, 3, 128, 128)
#     for i in range(5):
#         canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
#     return canvas
resize_width=transforms.Resize((width,width))
resize_128=transforms.Resize((128,128))
painter=ResNet(6, 18, 5*(param_num+3))
painter.load_state_dict(torch.load('model3/Paint-run1/actor.pkl'))
painter.eval()
# def decode(box, canvas, tar_canvas,debug=False,step=0):  # b * (10 + 3)
#     resize_128=transforms.Resize((128,128))
#     tar_canvas_box = []
#     canvas_box = []
#     ori_canvas=canvas.clone()
#     canvas=resize_128(canvas)
#     ori_tar_canvas=tar_canvas.clone()
#     tar_canvas=resize_128(tar_canvas)
#     for i in range(canvas.size(0)):
#         x1, y1, x2, y2 = torch.round(box[i]*127).detach().int()
#         x1, x2 = min(x1, x2), max(x1, x2)
#         y1, y2 = min(y1, y2), max(y1, y2)
#         resize = transforms.Resize((4 * (x2 + 1 - x1), 4 * (y2 + 1 - y1)))
#         #print(int(x1),int(x2),int(y1),int(y2))
#         if debug and i==0:
#             print('training box',int(x1),int(x2),int(y1),int(y2))
#         tar_canvas_box=resize_128(tar_canvas[i, :, x1:x2 + 1, y1:y2 + 1]).unsqueeze(0)
#         canvas_box=resize_128(canvas[i, :, x1:x2 + 1, y1:y2 + 1]).unsqueeze(0)
#         for kk in range(4):
#             param = painter(torch.cat((canvas_box, tar_canvas_box), dim=1))
#             x = param.view(-1, param_num + 3)
#             foregrounds, alphas, _ = Decoder(x[:, :param_num + 3])
#             foregrounds = foregrounds.view(-1, 5, 3, 128, 128)
#             alphas = alphas.view(-1, 5, 1, 128, 128)
#             #foregrounds[0] = morphology.dilation(foregrounds[0])
#             #alphas[0]=morphology.erosion(alphas[0])
#             for j in range(5):
#                 ori_canvas[i, :, 4*x1:4*(x2+1), 4*y1:4*y2+4] = \
#                     ori_canvas[i, :, 4*x1:4*(x2+1), 4*y1:4*y2+4] * resize(1 - alphas[0, j]) \
#                     + resize(alphas[0, j]) * resize(foregrounds[0, j])
#     return ori_canvas
def decode(box, canvas, tar_canvas,debug=False,step=0):  # b * (10 + 3) 128size的decode
    with torch.no_grad():
        tar_canvas_box = []
        canvas_box = []
        for i in range(canvas.size(0)):
            x1, y1, x2, y2 = torch.round(box[i]*(width-1)).detach().int()
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            # if debug and i==0:
            #     print(int(x1),int(x2),int(y1),int(y2))
            # if not debug and random.random()<0.001:
            #     print('training box',int(x1),int(x2),int(y1),int(y2))
            tar_canvas_box.append(resize_width(tar_canvas[i, :, x1:x2 + 1, y1:y2 + 1]))
            canvas_box.append(resize_width(canvas[i, :, x1:x2 + 1, y1:y2 + 1]))
        tar_canvas_box = torch.stack(tar_canvas_box, dim=0)
        canvas_box = torch.stack(canvas_box, dim=0)
        for kk in range(4):
            param = painter(torch.cat((canvas_box, tar_canvas_box), dim=1))
            x = param.view(-1, param_num + 3)
            foregrounds, alphas, _ = Decoder(x[:, :param_num + 3])
            foregrounds = foregrounds.view(-1, 5, 3, width, width)
            alphas = alphas.view(-1, 5, 1, width, width)
            for i in range(5):
                canvas_box = canvas_box * (1 - alphas[:, i]) + alphas[:, i] * foregrounds[:, i]
        for i in range(canvas.size(0)):
            x1, y1, x2, y2 = torch.round(box[i]*127).detach().int()
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            resize = transforms.Resize((x2+1 - x1, y2+1 - y1))
            canvas[i, :, x1:x2+1, y1:y2+1] = resize(canvas_box[i])
        return canvas
# def decode(box, canvas, tar_canvas,debug=False,step=0):  # b * (10 + 3)加红框的decode
#     save_canvas=canvas.clone()
#     with torch.no_grad():
#         tar_canvas_box = []
#         canvas_box = []
#         for i in range(canvas.size(0)):
#             x1, y1, x2, y2 = torch.round(box[i]*127).detach().int()
#             x1, x2 = min(x1, x2), max(x1, x2)
#             y1, y2 = min(y1, y2), max(y1, y2)
#             if debug and i==0:
#                 print(int(x1),int(x2),int(y1),int(y2))
#             # if not debug and random.random()<0.001:
#             #     print('training box',int(x1),int(x2),int(y1),int(y2))
#             tar_canvas_box.append(resize_128(tar_canvas[i, :, x1:x2 + 1, y1:y2 + 1]))
#             canvas_box.append(resize_128(canvas[i, :, x1:x2 + 1, y1:y2 + 1]))
#         tar_canvas_box = torch.stack(tar_canvas_box, dim=0)
#         canvas_box = torch.stack(canvas_box, dim=0)
#         for kk in range(4):
#             param = painter(torch.cat((canvas_box, tar_canvas_box), dim=1))
#             x = param.view(-1, param_num + 3)
#             foregrounds, alphas, _ = Decoder(x[:, :param_num + 3])
#             foregrounds = foregrounds.view(-1, 5, 3, 128, 128)
#             alphas = alphas.view(-1, 5, 1, 128, 128)
#             for i in range(5):
#                 canvas_box = canvas_box * (1 - alphas[:, i]) + alphas[:, i] * foregrounds[:, i]
#             for i in range(canvas.size(0)):
#                 x1, y1, x2, y2 = torch.round(box[i]*127).detach().int()
#                 x1, x2 = min(x1, x2), max(x1, x2)
#                 y1, y2 = min(y1, y2), max(y1, y2)
#                 resize = transforms.Resize((x2+1 - x1, y2+1 - y1))
#                 canvas[i, :, x1:x2+1, y1:y2+1] = resize(canvas_box[i])
#                 save_canvas[i]=canvas[i].clone()
#                 save_canvas[i,:,x1:x2+1,y1]=0
#                 save_canvas[i, :, x1:x2+1 + 1, y2] = 0
#                 save_canvas[i, :, x1, y1:y2+1] = 0
#                 save_canvas[i, :, x2, y1:y2 + 1] = 0
#                 save_canvas[i, 2, x1:x2 + 1, y1] = 1
#                 save_canvas[i, 2, x1:x2 + 1, y2] = 1
#                 save_canvas[i, 2, x1, y1:y2 + 1] = 1
#                 save_canvas[i, 2, x2, y1:y2 + 1] = 1
#             save_canvas=save_canvas[:,[2,1,0]]
#             save_image(save_canvas[:16],'test-images/%d-%d.png'%(step,kk),nrow=4,normalize=False)
#         return canvas
def cal_trans(s, t):
    return (s.transpose(0, 3) * t).transpose(0, 3)

class DDPG(object):
    def __init__(self, args,batch_size=64, env_batch=1, max_step=40, \
                 tau=0.001, discount=0.9, rmsize=800, \
                 writer=None, resume=None, output_path=None):
        self.args=args
        if args.style:
            self.style_reward=StyleReward()
        self.max_step = max_step
        self.env_batch = env_batch
        self.batch_size = batch_size
        self.actor = ResNet(6, 18, 4) # canvas,target
        self.actor_target = ResNet(6, 18, 4)
        self.critic = ResNet_wobn(10, 18, 1) # add the last canvas for better prediction
        self.critic_target = ResNet_wobn(10, 18, 1)
        self.actor_optim  = Adam(self.actor.parameters(), lr=1e-2)
        self.critic_optim  = Adam(self.critic.parameters(), lr=1e-2)

        if (resume != None):
            self.load_weights(resume)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        # Create replay buffer
        self.memory = rpm(rmsize * max_step)

        # Hyper-parameters
        self.tau = tau
        self.discount = discount

        # Tensorboard
        self.writer = writer
        self.log = 0

        self.state = [None] * self.env_batch # Most recent state
        self.action = [None] * self.env_batch # Most recent action
        self.choose_device()
        self.size=1

    def play(self, state, target=False,debug=False):
        state = state[:, :6].float() / 255
        if target:
            return self.actor_target(state)
        else:
            if debug:
                tmp=self.actor(state)
                print(tmp[0])
            return self.actor(state)

    def update_gan(self, state):
        canvas = state[:, :3]
        gt = state[:, 3 : 6]
        fake, real, penal = update(canvas.float() / 255, gt.float() / 255)
        # if self.log % 20 == 0:
        #     self.writer.add_scalar('train/gan_fake', fake, self.log)
        #     self.writer.add_scalar('train/gan_real', real, self.log)
        #     self.writer.add_scalar('train/gan_penal', penal, self.log)

    # def evaluate(self, state, action, target=False):
    #     T = state[:, 6 : 7]
    #     gt = state[:, 3 : 6].float() / 255
    #     canvas0 = state[:, :3].float() / 255
    #     canvas1 = decode(action, canvas0,gt)
    #     gan_reward = ((canvas0 - gt) ** 2).mean(1).mean(1).mean(1) - ((canvas1 - gt) ** 2).mean(1).mean(1).mean(1)
    #     action=action.unsqueeze(-1).unsqueeze(-1).repeat(1,1,128,128)
    #     merged_state = torch.cat([canvas0,gt,action], 1)
    #
    #     # canvas0 is not necessarily added
    #     if target:
    #         Q = self.critic_target(merged_state)
    #         return (Q + gan_reward), gan_reward
    #     else:
    #         Q = self.critic(merged_state)
    #         # if self.log % 20 == 0:
    #         #     self.writer.add_scalar('train/expect_reward', Q.mean(), self.log)
    #         #     self.writer.add_scalar('train/gan_reward', gan_reward.mean(), self.log)
    #         return (Q + gan_reward), gan_reward
    def test(self):
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([128,128])
        ])
        img=Image.open('/home/huteng/LearningToPaint-master/image/1.jpg')
        img=transform(img).unsqueeze(0)
        canvas=torch.zeros_like(img)
        state=torch.cat((canvas,img),1).cuda()*255
        # action=self.play(state)
        # Q,R=self.evaluate(state,action)
        # print('reward',Q,R)
        action=torch.tensor([[0.0,0.0,1.0,1.0]]).cuda()
        Q, R = self.evaluate(state, action)
        print('reward',Q, R)
    def f(self,x,alpha=0.1):
        return alpha*torch.log((1+x)/(1-x))
    def evaluate(self, state, action, target=False):
        gt = state[:, 3 : 6].float() / 255
        canvas0 = state[:, :3].float() / 255
        action_repeat=action.unsqueeze(-1).unsqueeze(-1).repeat(1,1,128,128)
        merged_state = torch.cat([resize_128(canvas0),resize_128(gt),action_repeat], 1)
        # canvas0 is not necessarily added
        if target:
            Q = self.critic_target(merged_state)
            return Q.squeeze(),None
        else:
            canvas1 = decode(action, canvas0.clone(), gt)
            dist0=((canvas0 - gt) ** 2).mean(1).mean(1).mean(1)
            if random.random() < 0.01:
                print(dist0.min(),dist0.mean())
            dist1=((canvas1 - gt) ** 2).mean(1).mean(1).mean(1)
            gan_reward=dist0-dist1
            alpha = (dist0 > 0.005).int()
            gan_reward=alpha*gan_reward+(1-alpha)*self.f(1-dist1)*gan_reward
            # alpha1=(dist0 > 0.005).int()
            # alpha2 = (dist0 > 0.003).int()
            # alpha3 = (dist0 > 0.001).int()
            # gan_reward=alpha1*gan_reward/4+(1-alpha1)*alpha2*gan_reward*125\
            #            +(1-alpha2)*alpha3*gan_reward*125+(1-alpha3)*gan_reward*250
            #gan_reward = alpha * gan_reward + (1 - alpha) * gan_reward
            Q = self.critic(merged_state)
            return Q.squeeze(),gan_reward
    def update_policy(self, lr):
        self.log += 1
        if self.log%1000==0 and self.log!=0:
            self.size*=0.95
        for param_group in self.critic_optim.param_groups:
            param_group['lr'] = lr[0]
        for param_group in self.actor_optim.param_groups:
            param_group['lr'] = lr[1]

        # Sample batch
        state, action, reward, \
            next_state, terminal = self.memory.sample_batch(self.batch_size, device)
        # if not self.args.style:
        #     self.update_gan(next_state)
        with torch.no_grad():
            next_action = self.play(next_state, True)
            target_q, _ = self.evaluate(next_state, next_action, True)
            target_q = self.discount * ((1 - terminal.float())) * target_q
        cur_q, step_reward = self.evaluate(state, action)
        target_q += step_reward.detach()
        #print(cur_q.shape,target_q.shape)
        value_loss = criterion(cur_q, target_q)
        self.critic.zero_grad()
        value_loss.backward(retain_graph=True)
        self.critic_optim.step()

        action = self.play(state)
        pre_q, _ = self.evaluate(state.detach(), action)
        #_, pre_q = self.evaluate(state.detach(), action)
        policy_loss = -pre_q.mean()
        self.actor.zero_grad()
        policy_loss.backward(retain_graph=True)
        #print(self.actor.conv1.weight.grad)
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return -policy_loss, value_loss

    def observe(self, reward, state, done, step):
        s0 = torch.tensor(self.state, device='cpu')
        a = to_tensor(self.action, "cpu")
        r = to_tensor(reward, "cpu")
        s1 = torch.tensor(state, device='cpu')
        d = to_tensor(done.astype('float32'), "cpu")
        for i in range(self.env_batch):
            self.memory.append([s0[i], a[i], r[i], s1[i], d[i]])
        self.state = state

    def noise_action(self, noise_factor, state, action):
        noise = np.zeros(action.shape)
        for i in range(self.env_batch):
            action[i] = action[i] + np.random.normal(0, self.noise_level[i], action.shape[1:]).astype('float32')
        return np.clip(action.astype('float32'), 0, 1)

    def select_action(self, state, return_fix=False, noise_factor=0,debug=False):
        self.eval()
        with torch.no_grad():
            action = self.play(state,debug=debug)
            action = to_numpy(action)
        if noise_factor > 0:
            if random.random()<noise_factor:
                action=np.random.rand(action.shape[0],action.shape[1])
            #action = self.noise_action(noise_factor, state, action)
        self.train()
        self.action = action
        if return_fix:
            return action
        return self.action

    def reset(self, obs, factor):
        self.state = obs
        self.noise_level = np.random.uniform(0, factor, self.env_batch)

    def load_weights(self, path):
        if path is None: return
        self.actor.load_state_dict(torch.load('{}/actor.pkl'.format(path)))
        self.critic.load_state_dict(torch.load('{}/critic.pkl'.format(path)))
        #load_gan(path)

    def save_model(self, path):
        self.actor.cpu()
        self.critic.cpu()
        torch.save(self.actor.state_dict(),'{}/actor.pkl'.format(path))
        torch.save(self.critic.state_dict(),'{}/critic.pkl'.format(path))
        save_gan(path)
        self.choose_device()

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def train(self):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()

    def choose_device(self):
        # self.actor=nn.DataParallel(self.actor).cuda()
        # self.actor_target=nn.DataParallel(self.actor_target).cuda()
        # self.critic=nn.DataParallel(self.critic).cuda()
        # self.critic_target=nn.DataParallel(self.critic_target).cuda()
        painter.to(device).eval()
        Decoder.to(device).eval()
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic.to(device)
        self.critic_target.to(device)
