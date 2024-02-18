import torch
import torch.nn as nn
import torch.nn.functional as F 

from collections import OrderedDict

class Expansion(nn.Module):
    def __init__(self, in_channel, kernel_size=5, pool=[4,4,5,5]):  
        super().__init__()
        
        padding = int((kernel_size - 1) // 2)
        
        self.conv_list = nn.ModuleList(
            [nn.Sequential(OrderedDict([
            ('conv',       nn.Conv1d(in_channel*(2**i), in_channel*(2**(i+1)), 
                                      kernel_size=kernel_size, padding=padding)),
            ('batch_norm', nn.BatchNorm1d(in_channel*(2**(i+1)))),
            ('relu',       nn.LeakyReLU()),
            ('pooling',    nn.AvgPool1d(p))
            ])) for i, p in enumerate(pool)])

        
    def forward(self, x):
        for layer in self.conv_list:
            x = layer(x)
        return x
    
class ResBlock(nn.Module):
    def __init__(self):
        super.__init__()
    def forward(self):
        pass

class ConvLSTM(nn.Module):
    def __init__(self, input_channel, hidden_channel, kernel_size, bidirectional=True):
        super().__init__()
        padding = int((kernel_size - 1) // 2)
        self.conv_layer = nn.Sequential(OrderedDict([
            ('conv1',       nn.Conv1d(input_channel, hidden_channel, 
                                      kernel_size=kernel_size, padding=padding)),
            ('batch_norm1', nn.BatchNorm1d(hidden_channel)),
            ('relu1',       nn.ReLU(hidden_channel)),
            ('pooling1',    nn.AvgPool1d(4)),
            
            ('conv2',       nn.Conv1d(hidden_channel, hidden_channel * 2,
                                      kernel_size=kernel_size, padding=padding)),
            ('batch_norm2', nn.BatchNorm1d(hidden_channel * 2)),
            ('relu2',       nn.ReLU(hidden_channel * 2)),
            ('pooling2',    nn.AvgPool1d(4)),
            
            ('conv3',       nn.Conv1d(hidden_channel * 2, hidden_channel * 4,
                                      kernel_size=kernel_size, padding=padding)),
            ('batch_norm3', nn.BatchNorm1d(hidden_channel * 4)),
            ('relu3',       nn.ReLU(hidden_channel * 4)),
            ('pooling3',    nn.AvgPool1d(5)),
            
            ('conv4',       nn.Conv1d(hidden_channel * 4, hidden_channel * 8,
                                      kernel_size=kernel_size, padding=padding)),
            ('batch_norm4', nn.BatchNorm1d(hidden_channel * 8)),
            ('relu4',       nn.ReLU(hidden_channel * 8)),
            ('pooling4',    nn.AvgPool1d(5))
        ]))
        self.lstm = nn.LSTM(hidden_channel * 8, hidden_channel * 8,
                             batch_first=True, bidirectional=bidirectional)
    def forward(self, x):
        
        out = self.conv_layer(x)
        out = out.permute(0,2,1)
        out, _ = self.lstm(out)
        return out