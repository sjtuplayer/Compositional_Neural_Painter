import cv2
import torch
import numpy as np
from env import Paint
from utils.util import *
from DRL.ddpg import decode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
from torchvision.utils import save_image
class fastenv():
    def __init__(self, 
                 max_episode_length, env_batch,log_dir, \
                 dataset_path,writer=None,style=False):
        self.max_episode_length = max_episode_length
        self.env_batch = env_batch
        self.env = Paint(self.env_batch, self.max_episode_length,dataset_path,style)
        self.env.load_data()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.writer = writer
        self.test = False
        self.log = 0
        self.log_dir=log_dir
    def save_img(self, log, step):
        canvas=self.env.canvas[:5].float()/255
        gt=self.env.gt[:5].float()/255
        canvas=canvas[:,[2,1,0]]
        gt=gt[:,[2,1,0]]
        #save_image(canvas,os.path.join(self.log_dir,'images/%d-%d.jpg'%(log,step)),nrow=5,normalized=False)
        save_image(canvas, os.path.join('../output', '%d-%d.jpg' % (log, step)), nrow=5, normalized=False)
        if step == self.max_episode_length:
            imgs=torch.cat((gt,canvas),dim=0)
            # save_image(imgs,os.path.join(self.log_dir,'images/%d.jpg'%log),nrow=5,normalized=False)
            save_image(imgs, os.path.join('../output', '%d.jpg' % log), nrow=5, normalized=False)
    def step(self, action,style=False,debug=False):
        with torch.no_grad():
            ob, r, d, _ = self.env.step(torch.tensor(action).to(device),style,debug)
        if d[0]:
            if not self.test:
                self.dist = self.get_dist()
                with open(os.path.join(self.log_dir,'train_loss.txt'),'a') as f:
                    f.write('log:%d,   l2_loss:%f\n'%(self.log,float(self.dist.mean())))
                    # self.writer.add_scalar('train/dist', self.dist[i], self.log)
                    self.log += 1
        return ob, r, d, _

    def get_dist(self):
        return to_numpy((((self.env.gt.float() - self.env.canvas.float()) / 255) ** 2).mean(1).mean(1).mean(1))
    def gen_style(self):
        loss,loss_content,loss_style=self.env.cal_dis(style=True,return_all=True)
        return to_numpy(loss),to_numpy(loss_content),to_numpy(loss_style)
    def reset(self, test=False, episode=0,style=False):
        self.test = test
        ob = self.env.reset(self.test, episode * self.env_batch,style)
        return ob
