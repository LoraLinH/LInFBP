
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
            self.weight = nn.Parameter(torch.Tensor(self.geo['nVoxelX']*self.geo['nVoxelY']*self.geo['views']*self.geo['extent']))
            self.bias = nn.Parameter(torch.Tensor(self.geo['nVoxelX']*self.geo['nVoxelY']))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, input):
        input = input.reshape(input.size(0), self.channel, self.geo['views']*self.geo['nDetecU'])

        indices = torch.round(self.geo['indices'])
        indices[indices>=self.geo['views']*self.geo['nDetecU']] = self.geo['views']*self.geo['nDetecU']-1
        output = torch.index_select(input, -1, indices.long())
        output = output.view(input.size(0), self.channel, -1, self.geo['views']*self.geo['extent'])

        return output

