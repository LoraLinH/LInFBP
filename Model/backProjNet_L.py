
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
        # nn.init.xavier_normal_(self.fc1.weight.data)
        # nn.init.constant_(self.fc1.bias, 0.01)
        # # nn.init.xavier_normal_(self.fc2.weight.data)
        # nn.init.constant_(self.fc2.weight, 0.01)
        # nn.init.constant_(self.fc2.bias, 0.01)
        # self.fc2.weight.data[0, ...] = 0.1

    def forward(self, input):

        x = self.fc1(input)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        # return x
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
        range_x = np.linspace(-1,1,self.range_x_num*2+1)*np.pi
        kernel = [np.ones_like(range_x), np.cos(range_x), np.sin(range_x),
                  np.cos(2*range_x), np.sin(2*range_x), np.cos(3*range_x), np.sin(3*range_x)]
        self.kernel = torch.tensor(kernel, dtype=torch.float)
        # self.project = SinoNet(channel, len(kernel))
        # self.project = Mlp(channel, len(kernel)*channel*2, (self.range_x_num*2+1) * channel)
        self.project = Mlp(channel, (self.range_x_num*2+1) * channel * 2, (self.range_x_num*2+1) * channel)
        self.register_parameter('weight', None)
        self.register_parameter('bias', None)

    def forward(self, input):

        input = input.permute(0,2,1,3).view(-1, self.channel, self.geo['nDetecU'])
        input = self.project(input.float())
        # input = input.view(-1, self.geo['views'], len(self.kernel)*self.channel, self.geo['nDetecU']).permute(0, 2, 1, 3)
        input = input.view(-1, self.geo['views'], (self.range_x_num*2+1) * self.channel, self.geo['nDetecU']).permute(0, 2, 1, 3)
        output = input.float().view(-1, self.channel, self.range_x_num * 2 + 1, self.geo['views'], self.geo['nDetecU']).permute(0, 1, 3, 4, 2)
        # 1 1 11 290 736

        # input = self.project(input.float())

        # input = input.float().view(-1, self.channel, len(self.kernel), self.geo['views'], self.geo['nDetecU'])
        # input = F.pad(input, pad=(self.range_x_num//2, self.range_x_num//2), mode="constant", value=0)
        # input = input.unfold(-1, self.range_x_num, 1)
        # 1 1 7 h w 5
        # output = torch.einsum('bachw, cd -> bahwd', input, self.kernel.to(input.device))

        output = output.flatten(-3)
        indices_low = torch.floor(self.geo['indices'])

        weight = torch.frac(self.geo['indices'])
        low_ind = indices_low * (2*self.range_x_num+1) + self.range_x_num + torch.floor(weight*self.range_x_num)
        low_weight = torch.frac(weight*self.range_x_num)
        high_ind = low_ind + self.range_x_num+1
        high_ind[high_ind>=output.size(-1)-1]=output.size(-1)-2
        radon_filtered_low1 = torch.index_select(output, -1, low_ind.long())
        radon_filtered_low2 = torch.index_select(output, -1, (low_ind+1).long())
        radon_filtered_low = radon_filtered_low1 * (1 - low_weight) + radon_filtered_low2 * low_weight
        radon_filtered_high1 = torch.index_select(output, -1, high_ind.long())
        radon_filtered_high2 = torch.index_select(output, -1, (high_ind+1).long())
        radon_filtered_high = radon_filtered_high1 * (1 - low_weight) + radon_filtered_high2 * low_weight
        output = radon_filtered_low * (1 - weight) + radon_filtered_high * weight

        output = output.view(-1, self.channel, self.geo['nVoxelX']*self.geo['nVoxelY'], self.geo['views']*self.geo['extent'])
        # output = torch.sum(output, 3) * (self.geo['end_angle']-self.geo['start_angle']) / (2*self.geo['views']*self.geo['extent'])
        # output = output.view(-1, self.channel, self.geo['nVoxelX'], self.geo['nVoxelY'])

        return output

