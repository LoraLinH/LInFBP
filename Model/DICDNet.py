import torch
import torch.nn as nn
import scipy.io as io
import torch.nn.functional as F
import numpy as np
from .backProjNet_linear import BackProjNet

kernel = io.loadmat('Results/init_kernel.mat')['C9']  # 3*32*9*9
kernel = torch.FloatTensor(kernel)
kernel = kernel[0:1, :, :, :]

filter = torch.FloatTensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]) / 9
filter = filter.unsqueeze(dim=0).unsqueeze(dim=0)



def iRadon_filter(sino, geo):
    npad = 2 ** np.ceil(np.log2(2 * geo['nDetecU'] - 1))  # padded size
    npad = int(npad)
    sino = torch.cat([sino, torch.zeros((sino.size(0), sino.size(1), geo['views'], npad - geo['nDetecU']), device=sino.device)], dim=-1)
    filter = geo['filter']
    _, _, H, W = sino.shape
    proj_fft = torch.fft.rfft(sino, dim=-1, norm='ortho')
    filter_proj = proj_fft * filter[:proj_fft.size(-1)]
    filter_sinogram = torch.fft.irfft(filter_proj, n=W, dim=-1, norm='ortho')
    filter_sinogram = filter_sinogram[...,:geo['nDetecU']]
    return filter_sinogram

class DICDNet(nn.Module):
    def __init__(self, geo, opt=None):
        super(DICDNet, self).__init__()
        self.geo = geo
        self.BackProjNet = BackProjNet(geo, geo['slices'])
        self.S = 10  # Stage number S includes the initialization process
        self.iter = self.S - 1  # not include the initialization process
        self.num_M = 32
        self.num_Q = 32  # for concatenation channel (See Supplementary material)

        self.for_flops = False

        # Stepsize
        self.etaM = torch.Tensor([1])  # initialization
        self.etaX = torch.Tensor([5])  # initialization
        self.etaM_S = self.make_eta(self.iter, self.etaM)
        self.etaX_S = self.make_eta(self.S, self.etaX)

        # kernel
        self.K0 = nn.Parameter(data=kernel, requires_grad=True)  # used in initialization process
        self.K = nn.Parameter(data=kernel, requires_grad=True)  # self.K (kernel) is inter-stage sharing

        # filter for initializing X and Q
        self.K_q_const = filter.expand(self.num_Q, 1, -1, -1).clone()  # size: self.num_Q*1*3*3
        self.K_q = nn.Parameter(self.K_q_const, requires_grad=True)
        self.fnet = Fnet()

        # proxNet
        self.proxNet_X_0 = Xnet()  # used in initialization process
        self.proxNet_X_S = self.make_Xnet(self.S)
        self.proxNet_M_S = self.make_Mnet(self.S)
        self.proxNet_X_last_layer = Xnet()  # fine-tune at the last

        # for sparsity
        self.tau_const = torch.Tensor([1])
        self.tau = nn.Parameter(self.tau_const, requires_grad=True)

    def make_Xnet(self, iters):
        layers = []
        for i in range(iters):
            layers.append(Xnet())
        return nn.Sequential(*layers)

    def make_Mnet(self, iters):
        layers = []
        for i in range(iters):
            layers.append(Mnet())
        return nn.Sequential(*layers)

    def make_eta(self, iters, const):
        const_dimadd = const.unsqueeze(dim=0)
        const_f = const_dimadd.expand(iters, -1).clone()
        eta = nn.Parameter(data=const_f, requires_grad=True)
        return eta

    def compute_gradient_proxy(self, FX, X):
        if self.for_flops:
            # FLOPs 模式：
            # 在这里额外运行 2 次 fnet，让 thop 累加这部分开销，从而模拟 gradient 的计算量
            self.fnet(X)
            self.fnet(X)
            return torch.zeros_like(X)
        else:
            # 正常模式：使用 autograd
            return torch.autograd.grad(outputs=FX, inputs=X, grad_outputs=torch.ones_like(X), retain_graph=True)[0]

    def forward(self, sino):
        sino = sino * self.geo['w1']
        filtered_sino = iRadon_filter(sino.float(), self.geo)
        output = self.BackProjNet(filtered_sino)
        output = output.reshape(sino.size(0), sino.size(1), self.geo['nVoxelX'], self.geo['nVoxelY'], self.geo['views'])
        output = output * self.geo['w2']
        output = torch.sum(output.float(), -1) * (self.geo['end_angle'] - self.geo['start_angle']) / (2 * self.geo['views'])
        fbp = output.reshape(sino.size(0), sino.size(1), self.geo['nVoxelX'], self.geo['nVoxelY'])
        # save mid-updating results
        ListX = []
        ListA = []
        ListM = []
        ListF = []
        Q00 = F.conv2d(fbp, self.K_q, stride=1, padding=1)  # dual variable Q (see supplementary material)
        input_ini = torch.cat((fbp, Q00), dim=1)
        XQ_ini = self.proxNet_X_0(input_ini)
        X0 = XQ_ini[:, :1, :, :]
        Q0 = XQ_ini[:, 1:, :, :]
        ListX.append(X0)

        # 1st iteration：Updating X0-->M1
        FX0 = self.fnet(X0)
        # grad_X = torch.autograd.grad(outputs=FX0, inputs=X0, grad_outputs=torch.ones_like(X0), retain_graph=True)[0]
        grad_X = self.compute_gradient_proxy(FX0, X0)

        A_hat = (fbp - X0 + FX0)
        A_hat_cut = F.relu(A_hat - self.tau)  # for sparsity
        Epsilon = F.conv_transpose2d(A_hat_cut, self.K0 / 10, stride=1, padding=4)  # /10 for controlling the updating speed
        M1 = self.proxNet_M_S[0](Epsilon)
        A = F.conv2d(M1, self.K / 10, stride=1, padding=4)  # /10 for controlling the updating speed

        # 1st iteration: Updating M1-->X1
        X_hat = fbp - A
        X_mid = (1 - self.etaX_S[0] / 10) * X0 + self.etaX_S[0] / 10 * X_hat * (grad_X-1)
        input_concat = torch.cat((X_mid, Q0), dim=1)
        XQ = self.proxNet_X_S[0](input_concat)
        X1 = XQ[:, :1, :, :]
        Q1 = XQ[:, 1:, :, :]
        ListX.append(X1)
        ListA.append(A)
        ListM.append(M1)
        ListF.append(FX0)
        X = X1
        Q = Q1
        M = M1
        for i in range(self.iter):
            # M-net
            FX = self.fnet(X)
            # grad_X = torch.autograd.grad(outputs=FX, inputs=X, grad_outputs=torch.ones_like(X), retain_graph=True)[0]
            grad_X = self.compute_gradient_proxy(FX0, X0)

            A_hat = (fbp - X + self.fnet(X))
            Epsilon = self.etaM_S[i, :] / 10 * F.conv_transpose2d((A - A_hat), self.K / 10, stride=1, padding=4)
            M = self.proxNet_M_S[i + 1](M - Epsilon)

            # X-net
            A = F.conv2d(M, self.K / 10, stride=1, padding=4)
            ListA.append(A)
            X_hat = fbp - A
            X_mid = (1 - self.etaX_S[i + 1, :]/ 10) * X + self.etaX_S[i + 1, :] / 10 * X_hat * (grad_X-1)
            input_concat = torch.cat((X_mid, Q), dim=1)
            XQ = self.proxNet_X_S[i + 1](input_concat)
            X = XQ[:, :1, :, :]
            Q = XQ[:, 1:, :, :]
            ListX.append(X)
            ListF.append(FX)
        XQ_adjust = self.proxNet_X_last_layer(XQ)
        X = XQ_adjust[:, :1, :, :]
        return X, ListX, ListA, ListF

    def inference(self, sino):
        with torch.no_grad():
            sino = sino * self.geo['w1']
            filtered_sino = iRadon_filter(sino.float(), self.geo)
            output = self.BackProjNet(filtered_sino)
            output = output.reshape(sino.size(0), sino.size(1), self.geo['nVoxelX'], self.geo['nVoxelY'], self.geo['views'])
            output = output * self.geo['w2']
            output = torch.sum(output.float(), -1) * (self.geo['end_angle'] - self.geo['start_angle']) / (2 * self.geo['views'])
            fbp = output.reshape(sino.size(0), sino.size(1), self.geo['nVoxelX'], self.geo['nVoxelY'])

            Q00 = F.conv2d(fbp, self.K_q, stride=1, padding=1)  # dual variable Q (see supplementary material)
            input_ini = torch.cat((fbp, Q00), dim=1)
            XQ_ini = self.proxNet_X_0(input_ini)
            X0 = XQ_ini[:, :1, :, :]
            Q0 = XQ_ini[:, 1:, :, :]

        # 1st iteration：Updating X0-->M1
        X0 = X0.requires_grad_()
        FX0 = self.fnet(X0)
        grad_X = torch.autograd.grad(outputs=FX0, inputs=X0, grad_outputs=torch.ones_like(X0), retain_graph=True)[0]

        with torch.no_grad():
            A_hat = (fbp - X0 + FX0)
            A_hat_cut = F.relu(A_hat - self.tau)  # for sparsity
            Epsilon = F.conv_transpose2d(A_hat_cut, self.K0 / 10, stride=1, padding=4)  # /10 for controlling the updating speed
            M1 = self.proxNet_M_S[0](Epsilon)
            A = F.conv2d(M1, self.K / 10, stride=1, padding=4)  # /10 for controlling the updating speed

            # 1st iteration: Updating M1-->X1
            X_hat = fbp - A
            X_mid = (1 - self.etaX_S[0] / 10) * X0 + self.etaX_S[0] / 10 * X_hat * (grad_X-1)
            input_concat = torch.cat((X_mid, Q0), dim=1)
            XQ = self.proxNet_X_S[0](input_concat)
            X1 = XQ[:, :1, :, :]
            Q1 = XQ[:, 1:, :, :]

        X = X1
        Q = Q1
        M = M1
        for i in range(self.iter):
            # M-net
            X = X.requires_grad_()
            FX = self.fnet(X)
            grad_X = torch.autograd.grad(outputs=FX, inputs=X, grad_outputs=torch.ones_like(X), retain_graph=True)[0]
            with torch.no_grad():
                A_hat = (fbp - X + self.fnet(X))
                Epsilon = self.etaM_S[i, :] / 10 * F.conv_transpose2d((A - A_hat), self.K / 10, stride=1, padding=4)
                M = self.proxNet_M_S[i + 1](M - Epsilon)

                # X-net
                A = F.conv2d(M, self.K / 10, stride=1, padding=4)
                X_hat = fbp - A
                X_mid = (1 - self.etaX_S[i + 1, :]/ 10) * X + self.etaX_S[i + 1, :] / 10 * X_hat * (grad_X-1)
                input_concat = torch.cat((X_mid, Q), dim=1)
                XQ = self.proxNet_X_S[i + 1](input_concat)
                X = XQ[:, :1, :, :]
                Q = XQ[:, 1:, :, :]
        with torch.no_grad():
            XQ_adjust = self.proxNet_X_last_layer(XQ)
            X = XQ_adjust[:, :1, :, :]
        return X

# proxNet_M
class Mnet(nn.Module):
    def __init__(self):
        super(Mnet, self).__init__()
        self.channels = 32
        self.T = 3  # the number of resblocks in each proxNet
        self.layer = self.make_resblock(self.T)
        self.tau0 = torch.Tensor([0.5])
        self.tau_const = self.tau0.unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0).expand(-1, self.channels, -1, -1).clone()
        self.tau = nn.Parameter(self.tau_const, requires_grad=True)  # for sparsity

    def make_resblock(self, T):
        layers = []
        for i in range(T):
            layers.append(
                nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                              nn.ReLU(),
                              nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                              ))
        return nn.Sequential(*layers)

    def forward(self, input):
        M = input
        for i in range(self.T):
            M = F.relu(M + self.layer[i](M))
        M = F.relu(M - self.tau)
        return M


# proxNet_X
class Xnet(nn.Module):
    def __init__(self):
        super(Xnet, self).__init__()
        self.channels = 32 + 1
        self.T = 3
        self.layer = self.make_resblock(self.T)

    def make_resblock(self, T):
        layers = []
        for i in range(T):
            layers.append(nn.Sequential(
                nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.ReLU(),
                nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
            ))
        return nn.Sequential(*layers)

    def forward(self, input):
        X = input
        for i in range(self.T-1):
            X = F.relu(X + self.layer[i](X))
        X = X + self.layer[-1](X)
        return X



class Fnet(nn.Module):
    def __init__(self):
        super(Fnet, self).__init__()
        self.channels = 32
        self.T = 3
        self.layer = self.make_resblock(self.T)

    def make_resblock(self, T):
        layers = []
        layers.append(nn.Sequential(
            nn.Conv2d(1, self.channels//2, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(self.channels//2, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
        ))
        for i in range(T-2):
            layers.append(nn.Sequential(
                nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.ReLU(),
                nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
            ))
        layers.append(nn.Sequential(
            nn.Conv2d(self.channels, self.channels // 2, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(self.channels // 2, 1, kernel_size=3, stride=1, padding=1, dilation=1),
        ))
        return nn.Sequential(*layers)

    def forward(self, input):
        X = self.layer(input)
        return X