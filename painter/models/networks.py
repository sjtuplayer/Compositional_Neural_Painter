import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            # lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            lr_l = 0.3 ** max(0, (epoch + opt.epoch_count - opt.n_epochs) // 5)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=()):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


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

class My_CNN_Painter(nn.Module):
    #输入当前canvas与目标图像，预测后八个笔触（全局）
    def __init__(self, param_per_stroke, total_strokes, n_heads=8, n_enc_layers=3, n_dec_layers=3):
        super().__init__()
        self.enc_img = models.vgg19(pretrained=True)
        self.conv = nn.Conv2d(6, 3, 3,1,1)
        hidden_dim=512*4*4
        self.linear_param = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//8),
            nn.ReLU(True),
            nn.Linear(hidden_dim//8, hidden_dim//64),
            nn.ReLU(True),
            nn.Linear(hidden_dim//64, param_per_stroke),
            nn.Sigmoid()
        )
        #self.linear_decider = nn.Linear(hidden_dim, 1)
        # self.query_pos = nn.Parameter(torch.rand(total_strokes, hidden_dim))
        # self.row_embed = nn.Parameter(torch.rand(8, hidden_dim // 2))
        # self.col_embed = nn.Parameter(torch.rand(8, hidden_dim // 2))
        self.resize=transforms.Resize([128,128])
    def forward(self, img, canvas):
        img=self.resize(img)
        canvas=self.resize(canvas)
        b, _, H, W = img.shape
        #feat = self.conv(torch.cat([img, canvas,num_layer.float()], dim=1))
        feat = self.conv(torch.cat([img, canvas], dim=1))
        feat = self.enc_img.features(feat)
        param=self.linear_param(feat.flatten(1)).unsqueeze(1)
        #param=param*0.9+0.05
        # print()
        # param[:,:,:4]=param[:,:,:4]*0.9+0.05
        return param
        # s=1
        # grid = param[:, :, :2].view(b * s, 1, 1, 2).contiguous()
        # #print(param[0, :, :2])
        # img_temp = img.unsqueeze(1).contiguous().repeat(1, s, 1, 1, 1).view(b * s, 3, H, W).contiguous()
        # color = nn.functional.grid_sample(img_temp, 2 * grid - 1, align_corners=False).view(b, s, 3).contiguous()
        # decision = self.linear_decider(hidden_state)
        #return torch.cat([param, color, color, torch.rand(b, s, 1, device=img.device)], dim=-1)
class My_Painter(nn.Module):
    #输入当前canvas与目标图像，预测后八个笔触（全局）
    def __init__(self, param_per_stroke, total_strokes, hidden_dim, n_heads=8, n_enc_layers=3, n_dec_layers=3):
        super().__init__()
        self.enc_img = models.vgg19(pretrained=True)
        self.conv = nn.Conv2d(512, hidden_dim, 1)
        self.transformer = nn.Transformer(hidden_dim, n_heads, n_enc_layers, n_dec_layers)
        self.linear_param = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=0.5),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=0.5),
            nn.ReLU(True),
            nn.Linear(hidden_dim, param_per_stroke),
            nn.Dropout(p=0.5),
            nn.Sigmoid())
        #self.linear_decider = nn.Linear(hidden_dim, 1)
        self.query_pos = nn.Parameter(torch.rand(total_strokes, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(8, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(8, hidden_dim // 2))

    def forward(self, img, canvas):
        b, _, H, W = img.shape
        feat = self.enc_img.features(img-canvas)
        h, w = feat.shape[-2:]
        feat_conv = self.conv(feat)

        pos_embed = torch.cat([
            self.col_embed[:w].unsqueeze(0).contiguous().repeat(h, 1, 1),
            self.row_embed[:h].unsqueeze(1).contiguous().repeat(1, w, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        hidden_state = self.transformer(pos_embed + feat_conv.flatten(2).permute(2, 0, 1).contiguous(),
                                        self.query_pos.unsqueeze(1).contiguous().repeat(1, b, 1))
        hidden_state = hidden_state.permute(1, 0, 2).contiguous()
        param = self.linear_param(hidden_state)
        s = hidden_state.shape[1]
        grid = param[:, :, :2].view(b * s, 1, 1, 2).contiguous()
        img_temp = img.unsqueeze(1).contiguous().repeat(1, s, 1, 1, 1).view(b * s, 3, H, W).contiguous()
        color = nn.functional.grid_sample(img_temp, 2 * grid - 1, align_corners=False).view(b, s, 3).contiguous()
        #decision = self.linear_decider(hidden_state)
        return torch.cat([param, color, color, torch.rand(b, s, 1, device=img.device)], dim=-1)

class Painter(nn.Module):

    def __init__(self, param_per_stroke, total_strokes, hidden_dim, n_heads=8, n_enc_layers=3, n_dec_layers=3,patch_size=32):
        super().__init__()
        self.enc_img = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, 3, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(True))
        self.enc_canvas = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, 3, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(True))
        self.conv = nn.Conv2d(128 * 2, hidden_dim, 1)
        self.transformer = nn.Transformer(hidden_dim, n_heads, n_enc_layers, n_dec_layers)
        self.linear_param = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, param_per_stroke))
        self.linear_decider = nn.Linear(hidden_dim, 1)
        self.query_pos = nn.Parameter(torch.rand(total_strokes, hidden_dim))
        self.row_embed2 = nn.Parameter(torch.rand(patch_size//4, hidden_dim // 2))
        self.col_embed2 = nn.Parameter(torch.rand(patch_size//4, hidden_dim // 2))

    def forward(self, img, canvas):
        b, _, H, W = img.shape
        img_feat = self.enc_img(img)
        canvas_feat = self.enc_canvas(canvas)
        h, w = img_feat.shape[-2:]
        feat = torch.cat([img_feat, canvas_feat], dim=1)
        feat_conv = self.conv(feat)
        pos_embed = torch.cat([
            self.col_embed2[:w].unsqueeze(0).contiguous().repeat(h, 1, 1),
            self.row_embed2[:h].unsqueeze(1).contiguous().repeat(1, w, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        hidden_state = self.transformer(pos_embed + feat_conv.flatten(2).permute(2, 0, 1).contiguous(),
                                        self.query_pos.unsqueeze(1).contiguous().repeat(1, b, 1))
        hidden_state = hidden_state.permute(1, 0, 2).contiguous()
        param = self.linear_param(hidden_state)
        s = hidden_state.shape[1]
        grid = param[:, :, :2].view(b * s, 1, 1, 2).contiguous()
        img_temp = img.unsqueeze(1).contiguous().repeat(1, s, 1, 1, 1).view(b * s, 3, H, W).contiguous()
        color = nn.functional.grid_sample(img_temp, 2 * grid - 1, align_corners=False).view(b, s, 3).contiguous()
        decision = self.linear_decider(hidden_state)
        return torch.cat([param, color, color, torch.rand(b, s, 1, device=img.device)], dim=-1), decision

class Style_Transferer(nn.Module):
    #输入当前canvas与目标图像，预测后八个笔触（全局）
    def __init__(self,  hidden_dim,total_strokes=8, n_heads=8, n_enc_layers=3, n_dec_layers=3):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.enc_img=nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            nn.Conv2d(1024, hidden_dim, 1)
            # self.resnet.layer4
        )
        self.resize=transforms.Resize((256,256))
        self.transformer = nn.Transformer(hidden_dim, n_heads, n_enc_layers, n_dec_layers)
        self.linear_param_hidden=nn.Sequential(
            nn.Linear(total_strokes, hidden_dim),
            nn.Dropout(p=0.5),
            nn.ReLU(True),
        )
        self.linear_img=nn.Sequential(
            nn.Linear(1024, 1328),
            nn.Dropout(p=0.5),
            nn.ReLU(True),
        )
        self.linear_param = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=0.5),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=0.5),
            nn.ReLU(True),
            nn.Linear(hidden_dim, total_strokes),
            nn.Dropout(p=0.5),
            nn.Sigmoid()
            #nn.Tanh()
        )
        #self.linear_decider = nn.Linear(hidden_dim, 1)
        #self.query_pos = nn.Parameter(torch.rand(total_strokes, hidden_dim))
        self.feat_pos=nn.Parameter(torch.rand(1,1328, hidden_dim))
        self.param_pos=nn.Parameter(torch.rand(1,1328, hidden_dim))
    def forward(self, old_param, img):
        param = self.linear_param_hidden(old_param)
        if img.size(-1) != 512 or img.size(-2) != 512:
            img = self.resize(img)
        feat = self.enc_img(img)
        b, _, H, W = img.shape
        feat=self.linear_img(feat.flatten(2)).permute(0,2,1)
        hidden_state = self.transformer((self.param_pos + param).permute(1, 0, 2),
                                        (self.feat_pos + feat).permute(1, 0, 2))
        hidden_state = hidden_state.permute(1, 0, 2)
        param = self.linear_param(hidden_state)
        # print(float(param.data.min()),float(param.data.max()))
        # print(float(old_param.data.min()),float(old_param.data.max()))
        param=(param-1)
        param=old_param+param
        #param=torch.clamp(old_param+param,0,1)
        print(float(param.data.min()),float(param.data.max()))
        return param
        #return torch.cat([param, param[:,:,-3:], torch.rand(b, 1024, 1, device=img.device)], dim=-1)
        # s = hidden_state.shape[1]
        # grid = param[:, :, :2].view(b * s, 1, 1, 2).contiguous()
        # img_temp = img.unsqueeze(1).contiguous().repeat(1, s, 1, 1, 1).view(b * s, 3, H, W).contiguous()
        # color = nn.functional.grid_sample(img_temp, 2 * grid - 1, align_corners=False).view(b, s, 3).contiguous()
        # #decision = self.linear_decider(hidden_state)
        # return torch.cat([param, color, color, torch.rand(b, s, 1, device=img.device)], dim=-1)
# if __name__ == '__main__':
#     net=Style_Transferer(256).cuda()
#     x1=torch.randn(2,1324,8).float().cuda()
#     x2=torch.randn(2,3,512,512).float().cuda()
#     y=net(x1,x2)
#     print(y.shape)
class PixelShuffleNet(nn.Module):
    def __init__(self, input_nc):
        super(PixelShuffleNet, self).__init__()
        self.fc1 = (nn.Linear(input_nc, 512))
        self.fc2 = (nn.Linear(512, 1024))
        self.fc3 = (nn.Linear(1024, 2048))
        self.fc4 = (nn.Linear(2048, 4096))
        self.conv1 = (nn.Conv2d(16, 32, 3, 1, 1))
        self.conv2 = (nn.Conv2d(32, 32, 3, 1, 1))
        self.conv3 = (nn.Conv2d(8, 16, 3, 1, 1))
        self.conv4 = (nn.Conv2d(16, 16, 3, 1, 1))
        self.conv5 = (nn.Conv2d(4, 8, 3, 1, 1))
        self.conv6 = (nn.Conv2d(8, 4*3, 3, 1, 1))
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = x.squeeze()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(-1, 16, 16, 16)
        x = F.relu(self.conv1(x))
        x = self.pixel_shuffle(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pixel_shuffle(self.conv6(x))
        x = x.view(-1, 3, 128, 128)
        return x
class DCGAN(nn.Module):
    def __init__(self, d, ngf=64):
        super(DCGAN, self).__init__()
        input_nc = d
        self.out_size = 128
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(input_nc, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4

            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf*2) x 64 x 64

            nn.ConvTranspose2d(ngf, 6, 4, 2, 1, bias=False),
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        output_tensor = self.main(input)
        return output_tensor[:,0:3,:,:], output_tensor[:,3:6,:,:]
class ZouFCNFusion(nn.Module):
    def __init__(self):
        super(ZouFCNFusion, self).__init__()
        self.out_size = 128
        self.huangnet = PixelShuffleNet(5)
        self.dcgan = DCGAN(12)

    def forward(self, x):
        x_shape = x[:, 0:5, :, :]
        x_alpha = x[:, [-1], :, :]
        x_alpha = torch.tensor(1.0)

        mask = self.huangnet(x_shape)
        color, _ = self.dcgan(x)

        return color * mask, x_alpha * mask
class FCN(nn.Module):
    def __init__(self,d=10,need_alphas=True,need_edge=True):
        super(FCN, self).__init__()
        self.need_alphas = need_alphas
        self.fc1 = (nn.Linear(d, 512))
        self.fc2 = (nn.Linear(512, 1024))
        self.fc3 = (nn.Linear(1024, 2048))
        self.fc4 = (nn.Linear(2048, 4096))
        self.conv1 = (nn.Conv2d(16, 32, 3, 1, 1))
        self.conv2 = (nn.Conv2d(32, 32, 3, 1, 1))
        self.conv3 = (nn.Conv2d(8, 16, 3, 1, 1))
        self.conv4 = (nn.Conv2d(16, 16, 3, 1, 1))
        self.conv5 = (nn.Conv2d(4, 8, 3, 1, 1))
        if need_edge:
            self.conv6 = (nn.Conv2d(8, 12, 3, 1, 1))
        elif need_alphas:
            self.conv6 = (nn.Conv2d(8, 8, 3, 1, 1))
        else:
            self.conv6 = (nn.Conv2d(8, 4, 3, 1, 1))
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.stylized = False
    def forward(self,x,need_edge=False):
        x=x.squeeze()
        tmp = 1 - self.draw(x[:, :-3])
        stroke = tmp[:, 0]
        alpha = tmp[:, 1]
        if need_edge:
            edge=tmp[:, 2]
        stroke = stroke.view(-1, 128, 128, 1)
        alpha = alpha.view(-1, 128, 128, 1)
        if need_edge:
            edge=edge.view(-1,128,128,1)
        color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
        alpha = alpha.permute(0, 3, 1, 2)
        color_stroke = color_stroke.permute(0, 3, 1, 2)
        if need_edge:
            edge=edge.permute(0,3,1,2)
        if need_edge:
            return color_stroke, alpha,edge
        else:
            return color_stroke, alpha
    def draw(self, x):
        b=x.size(0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(-1, 16, 16, 16)
        x = F.relu(self.conv1(x))
        x = self.pixel_shuffle(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pixel_shuffle(self.conv6(x))
        x = torch.sigmoid(x)
        return 1 - x.view(b,-1, 128, 128).squeeze()
def conv3x3(in_planes, out_planes, stride=1):
    return (nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False))

def cfg(depth):
    depth_lst = [18, 34, 50, 101, 152]
    assert (depth in depth_lst), "Error : Resnet depth should be either 18, 34, 50, 101, 152"
    cf_dict = {
        '18': (BasicBlock, [2,2,2,2]),
        '34': (BasicBlock, [3,4,6,3]),
        '50': (Bottleneck, [3,4,6,3]),
        '101':(Bottleneck, [3,4,23,3]),
        '152':(Bottleneck, [3,8,36,3]),
    }

    return cf_dict[str(depth)]
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                (nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = (nn.Conv2d(in_planes, planes, kernel_size=1, bias=False))
        self.conv2 = (nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
        self.conv3 = (nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False))
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                (nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out
class ResNet(nn.Module):
    def __init__(self, num_inputs, depth, num_outputs,pool_factor=4):
        super(ResNet, self).__init__()
        self.in_planes = 64

        block, num_blocks = cfg(depth)

        self.conv1 = conv3x3(num_inputs, 64, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_outputs)
        self.pool_factor=pool_factor
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)
    def forward2(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, self.pool_factor)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x
    def forward(self, target,canvas,step=None):
        x=torch.cat((target,canvas),dim=1)
        if step is not None:
            tmp=torch.ones(target.size(0),1,target.size(-2),target.size(-1)).cuda().float()*step
            x=torch.cat((x,tmp),dim=1)
        #x = F.relu(self.bn1(self.conv1(x[:,:6])))
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, self.pool_factor)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x
class ResNet_box(nn.Module):
    def __init__(self, num_inputs, depth, num_outputs):
        super(ResNet_box, self).__init__()
        self.in_planes = 64

        block, num_blocks = cfg(depth)

        self.conv1 = conv3x3(num_inputs, 64, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_outputs)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, target,canvas,step):
        x=torch.cat((target,canvas),dim=1)
        tmp=torch.ones(target.size(0),1,target.size(-2),target.size(-1)).cuda().float()*step
        x=torch.cat((x,tmp),dim=1)
        #x = F.relu(self.bn1(self.conv1(x[:,:6])))
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x