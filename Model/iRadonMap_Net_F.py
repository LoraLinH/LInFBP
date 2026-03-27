
import torch.nn as nn
from Utils.initParameter import InitPara
# import astra
from torch.autograd import Function
import numpy as np
import torch
import copy
from .backProjNet_F import BackProjNet

def iRadon(sino, geo):
    npad = 2 ** np.ceil(np.log2(2 * geo['nDetecU'] - 1))  # padded size
    npad = int(npad)
    sino = torch.cat([sino, torch.zeros((1, 1, geo['views'], npad - geo['nDetecU'])).cuda()], dim=-1)
    filter = geo['filter']
    proj_fft = torch.fft.rfft(sino, dim=-1, norm='ortho')
    filter_proj = proj_fft * filter[:npad//2+1]
    filter_sinogram = torch.fft.irfft(filter_proj, n=npad, dim=-1, norm='ortho')
    filter_sinogram = filter_sinogram[...,:geo['nDetecU']]
    return filter_sinogram

def int2tensor(input):
    output = np.array(input)
    output = torch.from_numpy(output)

    return output

class ResidualBlock(nn.Module):
    def __init__(self, planes):
        super(ResidualBlock, self).__init__()

        self.filter1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.ln1 = nn.GroupNorm(num_channels=planes, num_groups=1, affine=False)
        self.leakyrelu1 = nn.LeakyReLU(0.2, True)
        self.filter2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.ln2 = nn.GroupNorm(num_channels=planes, num_groups=1, affine=False)
        self.leakyrelu2 = nn.LeakyReLU(0.2, True)

    def forward(self, input):
        output = self.leakyrelu1(self.ln1(self.filter1(input)))
        output = self.ln2(self.filter2(output))
        output += input
        output = self.leakyrelu2(output)

        return output

class SpatialNet(nn.Module):
    def __init__(self, opt):
        super(SpatialNet, self).__init__()

        self.planes = opt.num_filters * 4

        model_list = [nn.Conv2d(1, self.planes, kernel_size=3, stride=1, padding=1, bias=True),
                      nn.GroupNorm(num_channels=4 * opt.num_filters, num_groups=1, affine=False),
                      nn.LeakyReLU(0.2, True)]
        model_list += [ResidualBlock(planes=self.planes), ResidualBlock(planes=self.planes)]
        model_list += [ResidualBlock(planes=self.planes), ResidualBlock(planes=self.planes)]
        model_list += [ResidualBlock(planes=self.planes), ResidualBlock(planes=self.planes)]
        model_list += [ResidualBlock(planes=self.planes), ResidualBlock(planes=self.planes)]
        model_list += [nn.Conv2d(4 * opt.num_filters, opt.num_filters, kernel_size=1, stride=1, padding=0, bias=True)]
        model_list += [nn.Conv2d(opt.num_filters, 1, kernel_size=1, stride=1, padding=0, bias=True)]

        self.model = nn.Sequential(*model_list)


    def forward(self, input):
        input = input.view(1, 1, 512, 512)
        output = self.model(input)
        return output

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class ResidualBlock_1d(nn.Module):
    def __init__(self, planes, reduction):
        super(ResidualBlock_1d, self).__init__()

        self.filter1 = nn.Conv2d(planes, planes, kernel_size=(1,3), stride=1, padding=(0,1), bias=True)
        self.ln1 = nn.GroupNorm(num_channels=planes, num_groups=1, affine=False)
        self.leakyrelu1 = nn.LeakyReLU(0.2, True)
        self.filter2 = nn.Conv2d(planes, planes, kernel_size=(1,3), stride=1, padding=(0,1), bias=True)
        self.ln2 = nn.GroupNorm(num_channels=planes, num_groups=1, affine=False)
        self.leakyrelu2 = nn.LeakyReLU(0.2, True)
        self.CA = CALayer(planes, reduction)

    def forward(self, input):
        output = self.leakyrelu1(self.ln1(self.filter1(input)))
        output = self.ln2(self.filter2(output))
        output = self.leakyrelu2(output)
        output = self.CA(output)

        return output + input

class SinoNet(nn.Module):
    def __init__(self, opt):
        super(SinoNet, self).__init__()

        self.planes = 4 * opt.num_filters
        conv_start = [nn.Conv2d(1, opt.num_filters, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True),
                      nn.GroupNorm(num_channels=opt.num_filters, num_groups=1, affine=False), nn.LeakyReLU(0.2, True)]
        conv_start += [nn.Conv2d(opt.num_filters, 4 * opt.num_filters, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True),
            nn.GroupNorm(num_channels=4 * opt.num_filters, num_groups=1, affine=False), nn.LeakyReLU(0.2, True)]
        self.model_start = nn.Sequential(*conv_start)
        self.filter1 = nn.Sequential(*[ResidualBlock_1d(4 * opt.num_filters, opt.reduction), ResidualBlock_1d(4 * opt.num_filters, opt.reduction), ResidualBlock_1d(4 * opt.num_filters, opt.reduction)])
        self.filter2 = nn.Sequential(*[ResidualBlock_1d(4 * opt.num_filters, opt.reduction), ResidualBlock_1d(4 * opt.num_filters, opt.reduction), ResidualBlock_1d(4 * opt.num_filters, opt.reduction)])
        self.filter3 = nn.Sequential(*[ResidualBlock_1d(4 * opt.num_filters, opt.reduction), ResidualBlock_1d(4 * opt.num_filters, opt.reduction), ResidualBlock_1d(4 * opt.num_filters, opt.reduction)])

        model_list_final = [nn.Conv2d(4 * opt.num_filters, opt.num_filters, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True)]
        model_list_final += [nn.Conv2d(opt.num_filters, 1, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True)]
        self.model_final = nn.Sequential(*model_list_final)

    def forward(self, input):
        output = self.model_start(input)
        output = self.filter1(output)
        output = self.filter2(output)
        output = self.filter3(output)
        output = self.model_final(output)
        return output

class iRadonMap_F(nn.Module):
    def __init__(self, geo, opt):
        super(iRadonMap_F, self).__init__()
        self.SinoNet = SinoNet(opt).cuda()
        self.BackProjNet = BackProjNet(geo, geo['slices']).cuda()
        self.SpatialNet = SpatialNet(opt).cuda()
        self.geo = geo

        
    def forward(self, input):

        output = self.SinoNet(input)
        output = output * self.geo['w1']
        output = iRadon(output.float(), self.geo)
        output = self.BackProjNet(output)
        output = output.reshape(input.size(0), input.size(1), self.geo['nVoxelX'], self.geo['nVoxelY'], self.geo['views'])
        output = output * self.geo['w2']
        output = torch.sum(output.float(), -1) * np.pi / self.geo['views']
        output = self.SpatialNet(output)

        return output
