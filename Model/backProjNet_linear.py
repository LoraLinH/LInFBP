
import torch 
import torch.nn as nn
from Model.utils import DotProduct 


class BackProjNet(nn.Module):
    def __init__(self, geo, channel=8, learn=False):
        super(BackProjNet, self).__init__()
        self.geo = geo
        self.learn = learn
        self.channel = channel
        
        if self.learn:
            self.weight = nn.Parameter(torch.Tensor(self.geo['nVoxelX'], self.geo['nVoxelY'], self.geo['views']*self.geo['extent']))
            self.bias = nn.Parameter(torch.Tensor(self.geo['nVoxelX'], self.geo['nVoxelY']))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, input):
        input = input.reshape(1, self.channel, self.geo['views']*self.geo['nDetecU'])
        indices = self.geo['indices'].view(self.geo['nVoxelX'], self.geo['nVoxelY'], self.geo['views'] * self.geo['extent'])
        indices_low = torch.floor(indices)
        indices_high = torch.ceil(indices)
        weight = torch.frac(indices)

        indices_high[indices_high == self.geo['views'] * self.geo['nDetecU']] = self.geo['views'] * self.geo['nDetecU'] - 1

        radon_filtered_low = torch.index_select(input, 2, indices_low.long().flatten())
        radon_filtered_high = torch.index_select(input, 2, indices_high.long().flatten())

        output = radon_filtered_low * (1 - weight.flatten()) + radon_filtered_high * weight.flatten()
        output = output.view(-1, self.channel, indices.size(0), indices.size(1), self.geo['views']*self.geo['extent'])


        return output

