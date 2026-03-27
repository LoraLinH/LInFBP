
import torch.nn as nn
from Model.backProjNet_F import BackProjNet
import numpy as np
import torch
from scipy.fftpack import fftshift, fftfreq

# def iRadon(sino, geo):
#     f = fftfreq(geo['nDetecU'])
#     fourier_filter = 2 * np.abs(f)
#     filter_ = torch.from_numpy(fourier_filter).to(sino.device)
#     _, _, H, W = sino.shape
#     proj_fft = torch.fft.rfft2(sino, dim=(-2, -1), norm='ortho')
#     filter_proj = proj_fft * filter_[:proj_fft.size(-1)]
#     filter_sinogram = torch.fft.irfft2(filter_proj, s=(H, W), dim=(-2, -1), norm='ortho')
#     return filter_sinogram

def iRadon(sino, geo):
    npad = 2 ** np.ceil(np.log2(2 * geo['nDetecU'] - 1))  # padded size
    npad = int(npad)
    # sino_size = (sino.size(0), sino.size(1), geo['views'], npad - geo['nDetecU'])
    sino_size = (1, 1, geo['views'], npad - geo['nDetecU'])
    sino = torch.cat([sino, torch.zeros(sino_size).cuda()], dim=-1)
    filter = geo['filter']
    # _, _, H, W = sino.shape
    proj_fft = torch.fft.rfft(sino, dim=-1, norm='ortho')
    filter_proj = proj_fft * (filter[:npad//2+1])

    # filter_proj = proj_fft * filter
    filter_sinogram = torch.fft.irfft(filter_proj, n=npad, dim=-1, norm='ortho')
    filter_sinogram = filter_sinogram[...,:geo['nDetecU']]
    return filter_sinogram


def iRadon2(sino, geo):
    s = geo['nDetecU']
    step = 2 * np.pi / s
    w = np.arange(-np.pi, np.pi, step)
    if len(w) < s:
        w = np.concatenate([w, w[-1] + step])
    r = np.abs(w)
    # r[:100] = 0
    # r[-100:] = 0
    filter_ = torch.from_numpy(fftshift(r)).to(sino.device)
    _, _, H, W = sino.shape
    proj_fft = torch.fft.rfft2(sino, dim=(-2, -1), norm='ortho')
    filter_proj = proj_fft * filter_[:proj_fft.size(-1)]
    filter_sinogram = torch.fft.irfft2(filter_proj, s=(H, W), dim=(-2, -1), norm='ortho')
    return filter_sinogram


class FBP_F(nn.Module):
    def __init__(self, geo, opt=None):
        super(FBP_F, self).__init__()
        self.geo = geo
        self.bp_channel = geo['slices']
        # self.param = nn.Conv2d(3,3,1)

        self.BackProjNet = BackProjNet(geo, self.bp_channel).cuda()

        
    def forward(self, input):
        output = input.cuda()
        output = output * self.geo['w1']
        output = iRadon(output.float(), self.geo)
        output = self.BackProjNet(output.float())
        w2 = self.geo['w2']
        output = output.reshape(input.size(0), input.size(1), self.geo['nVoxelX'], self.geo['nVoxelY'], self.geo['views'] * self.geo['extent'])
        output = output * w2
        output = torch.sum(output.float(), -1) * np.pi / (self.geo['views'] * self.geo['extent'])
        output = output.reshape(input.size(0), input.size(1), self.geo['nVoxelX'], self.geo['nVoxelY'])
        return output



