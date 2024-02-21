import torch
import torch.nn as nn
import torch.nn.functional as F 

from collections import OrderedDict

class Expansion(nn.Module):
    def __init__(self, in_channel, kernel_size, pool, alpha):  
        super().__init__()
        
        padding = [int((k - 1) // 2) for k in kernel_size]
        
        self.conv_list = nn.ModuleList(
            [nn.Sequential(OrderedDict([
            ('conv',       nn.Conv1d(in_channel*int(alpha**i), in_channel*int(alpha**(i+1)), 
                                      kernel_size=k, padding=pad)),
            ('batch_norm', nn.BatchNorm1d(in_channel*int(alpha**(i+1)))),
            ('relu',       nn.LeakyReLU()),
            ('pooling',    nn.AvgPool1d(pool))
            ])) for i, (k,pad, pool) in enumerate(zip(kernel_size, padding, pool))])

        
    def forward(self, x):
        for layer in self.conv_list:
            x = layer(x)
        return x

class Expansion_2d(nn.Module):
    def __init__(self, in_feature, in_channel, kernel_size=[5,5,5,5,5],
                 pool_channel=[4,4,5,5,5], alpha=2):  
        super().__init__()
        
        padding = [int((k - 1) // 2) for k in kernel_size]
        
        expand_layer1 = nn.ModuleList(
            [nn.Sequential(OrderedDict([
            ('conv',       nn.Conv1d(in_channel*int(alpha**i), in_channel*int(alpha**(i+1)), 
                                      kernel_size=k, padding=pad)),
            ('batch_norm', nn.BatchNorm1d(in_channel*int(alpha**(i+1)))),
            ('relu',       nn.LeakyReLU()),
            ('pooling',    nn.AvgPool1d(pool))
            ])) for i, (k,pad, pool) in enumerate(zip(kernel_size, padding, pool_channel))])
        
        self.expand_bin = nn.ModuleList([expand_layer1 for i in range(in_feature)])
        
        
    def forward(self, x):
        B, L, M, N = x.shape

        outs = []
        for i, layer in enumerate(self.expand_bin):
            _out = x[:,i].permute(0,2,1)
            for j, elem in enumerate(layer):
                _out = elem(_out)
            _out = F.avg_pool1d(_out, _out.shape[-1]).squeeze()
            outs.append(_out)
        
        out = torch.stack(outs).permute(1,0,2)
        return out

class Attention1d(nn.Module):
    def __init__(self, kernel_size=[5,5]):
        super().__init__()
        padding = [int((k - 1) // 2) for k in kernel_size]
        self.attention = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(1,1,kernel_size=kernel_size[0], padding=padding[0])),
            ('relu',  nn.LeakyReLU()),
            ('conv2', nn.Conv1d(1,1,kernel_size=kernel_size[1], padding=padding[1]))]))
        
    def forward(self, x):
        attention = self.attention(x)
        attention -= attention.min(1, keepdim=True)[0]
        attention /= attention.max(1, keepdim=True)[0]
        
        out = x * attention
        return out

class Waveform_Attention(nn.Module):
    def __init__(self, in_channel, kernel_size=[5,5]):
        super().__init__()
        self.attentions = nn.ModuleList([
            Attention1d(kernel_size=kernel_size) for i in range(in_channel)
        ])
    def forward(self, x):
        B, L, M = x.shape
        x = x.permute(1,0,2)
        outs = []
        for eeg, layer in zip(x, self.attentions):
            eeg = eeg.unsqueeze(1)
            out = layer(eeg)
            out = out.squeeze(1)
            outs.append(out)
        outs = torch.stack(outs)
        outs = outs.permute(1,0,2)
        return outs
    
class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
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
    
#TEST
if __name__ == "__main__":
    x = torch.rand(32,20,10000)
    model = Waveform_Attention(in_channel=20)
    y = model(x)
    print(y.shape)