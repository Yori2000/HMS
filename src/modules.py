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
            ('conv2', nn.Conv1d(1,1,kernel_size=kernel_size[1], padding=padding[1])),
            ('sigmoid', nn.Sigmoid())]))
        
    def forward(self, x):
        attention = self.attention(x)
        
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
    
    
class WaveBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates, kernel_size):
        super(WaveBlock, self).__init__()
        
        self.num_rates= dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True))
        
        dilation_rates = [2 ** i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                          padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))
            self.gate_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                          padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=True))
            
        for i in range(len(self.convs)):
            nn.init.xavier_uniform_(self.convs[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.convs[i].bias)

        for i in range(len(self.filter_convs)):
            nn.init.xavier_uniform_(self.filter_convs[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.filter_convs[i].bias)

        for i in range(len(self.gate_convs)):
            nn.init.xavier_uniform_(self.gate_convs[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.gate_convs[i].bias)
            
    def forward(self, x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            tanh_out = torch.tanh(self.filter_convs[i](x))
            sigmoid_out = torch.sigmoid(self.gate_convs[i](x))
            x = tanh_out * sigmoid_out
            x = self.convs[i + 1](x) 
            res = res + x
        return res

class WaveNet(nn.Module):
    def __init__(self, input_channels, kernel_size, dilation_rates):
        super(WaveNet, self).__init__()
        self.model = nn.Sequential(
                WaveBlock(input_channels, 8, dilation_rates[0], kernel_size),
                WaveBlock(8, 16, dilation_rates[1], kernel_size),
                WaveBlock(16, 32, dilation_rates[2], kernel_size),
                WaveBlock(32, 64, dilation_rates[3], kernel_size) 
        )
    def forward(self, x):
        output = self.model(x)
        return output
    
#TEST
if __name__ == "__main__":
    x = torch.rand(32,20,10000)
    model = Waveform_Attention(in_channel=20)
    y = model(x)
    print(y.shape)