
import numpy as np
import torch 
import torch.nn as nn
from Model.utils import DotProduct
from torch.nn import functional as F

from Model.utils import ResidualBlock





class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv1d(in_features, hidden_features, kernel_size=3, padding=1)
        self.act = act_layer()
        self.fc2 = nn.Conv1d(hidden_features, out_features, kernel_size=3, padding=1)
        self.drop = nn.Dropout(drop)
        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.constant_(self.fc1.bias, 0.01)
        # nn.init.xavier_normal_(self.fc2.weight.data)
        nn.init.constant_(self.fc2.weight, 0.01)
        nn.init.constant_(self.fc2.bias, 0.01)
        self.fc2.weight.data[0, ...] = 0.1

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

# class SinoNet(nn.Module):
#     def __init__(self, bp_channel, num_filters):
#         super(SinoNet, self).__init__()
#         model_list = [nn.Conv2d(bp_channel, num_filters, kernel_size=3, stride=1, padding=1, bias=True),
#                       nn.GroupNorm(num_channels=num_filters, num_groups=1, affine=False), nn.LeakyReLU(0.2, True)]
#         model_list += [nn.Conv2d(num_filters, num_filters*4, kernel_size=3, stride=1, padding=1, bias=True),
#                       nn.GroupNorm(num_channels=num_filters*4, num_groups=1, affine=False), nn.LeakyReLU(0.2, True)]
#         # model_list += [ResidualBlock(planes=num_filters * 4), ResidualBlock(planes=num_filters * 4),
#         #                ResidualBlock(planes=num_filters * 4)]
#         # model_list += [ResidualBlock(planes=num_filters * 4), ResidualBlock(planes=num_filters * 4),
#         #                ResidualBlock(planes=num_filters * 4)]
#         # model_list += [ResidualBlock(planes=num_filters * 4), ResidualBlock(planes=num_filters * 4),
#         #                ResidualBlock(planes=num_filters * 4)]
#         # model_list += [ResidualBlock(planes=num_filters * 4), ResidualBlock(planes=num_filters * 4),
#         #                ResidualBlock(planes=num_filters * 4)]
#
#         model_list += [nn.Conv2d(num_filters * 4, num_filters, kernel_size=3, stride=1, padding=1, bias=True)]
#         self.model = nn.Sequential(*model_list)
#
#     def forward(self, input):
#         return self.model(input)

# class ConvMlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=3, padding=1)
#         self.act = act_layer()
#         self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=3, padding=1)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x

class BackProjNet(nn.Module):
    def __init__(self, geo, channel=8, learn=False):
        super(BackProjNet, self).__init__()
        self.geo = geo
        self.learn = learn
        self.channel = channel
        self.range_x_num = 5
        range_x = np.linspace(-1,1,self.range_x_num)
        kernel = [np.ones_like(range_x), np.cos(range_x), np.sin(range_x),
                  np.cos(2*range_x), np.sin(2*range_x), np.cos(3*range_x), np.sin(3*range_x)]
        # kernel = [np.ones_like(range_x), np.cos(range_x), np.sin(range_x),
        #                     np.cos(2*range_x), np.sin(2*range_x)]
        self.kernel = torch.tensor(kernel, dtype=torch.float)
        # self.project = SinoNet(channel, len(kernel))
        self.project = Mlp(channel, len(kernel)*channel*2, len(kernel)*channel)
        self.register_parameter('weight', None)
        self.register_parameter('bias', None)


    def forward(self, input, crop_size=None):

        input = input.permute(0,2,1,3).view(-1, self.channel, self.geo['nDetecU'])
        input = self.project(input.float())
        input = input.view(-1, self.geo['views'], len(self.kernel)*self.channel, self.geo['nDetecU']).permute(0, 2, 1, 3)
        # input = self.project(input.float())
        input = input.float().reshape(-1, self.channel, len(self.kernel), self.geo['views']*self.geo['nDetecU'])
        if crop_size is not None:
            indices = self.geo['indices'].reshape(self.geo['nVoxelX'], self.geo['nVoxelY'], self.geo['views'])
            indices = indices[crop_size[0,0]:crop_size[0,0]+self.geo['crop_size'],crop_size[0,1]:crop_size[0,1]+self.geo['crop_size']]
            indices = indices.flatten()
        else:
            indices = self.geo['indices']
        indices_low = torch.floor(indices)
        indices_high = torch.ceil(indices)
        indices_high[indices_high == self.geo['views'] * self.geo['nDetecU']] = self.geo['views'] * self.geo['nDetecU'] - 1
        weight = torch.frac(indices)
        radon_filtered_low = torch.index_select(input, -1, indices_low.long())
        radon_filtered_high = torch.index_select(input, -1, indices_high.long())
        kernel_low = torch.stack([torch.ones_like(weight), torch.cos(weight), torch.sin(weight), torch.cos(2 * weight),
                                  torch.sin(2 * weight), torch.cos(3 * weight), torch.sin(3 * weight)])
        kernel_high = torch.stack([torch.ones_like(weight - 1.), torch.cos(weight - 1.), torch.sin(weight - 1.), torch.cos(2 * weight - 2.),
             torch.sin(2 * weight - 2.), torch.cos(3 * weight - 3.), torch.sin(3 * weight - 3.)])
        # kernel_low = torch.stack([torch.ones_like(weight), torch.cos(weight), torch.sin(weight), torch.cos(2 * weight), torch.sin(2 * weight)])
        # kernel_high = torch.stack([torch.ones_like(weight - 1.), torch.cos(weight - 1.), torch.sin(weight - 1.), torch.cos(2 * weight - 2.), torch.sin(2 * weight - 2.)])
        output_low = torch.einsum('bacn, cn -> ban', radon_filtered_low, kernel_low.to(input.device))
        output_high = torch.einsum('bacn, cn -> ban', radon_filtered_high, kernel_high.to(input.device))
        output_tmp = output_low * (1 - weight) + output_high * weight
        output = output_tmp.view(input.size(0), self.channel, -1, self.geo['views'] * self.geo['extent'])
        return output



