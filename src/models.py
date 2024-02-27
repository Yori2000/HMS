import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from modules import Expansion, Expansion_2d, Waveform_Attention
from utils import Config

import hydra
from collections import OrderedDict
from omegaconf import DictConfig
from logging import getLogger
import math

logger = getLogger('main').getChild('model')


def get_model(cfg):
    models = []

    if cfg.model_name =='Conv_Flatten':
        return Conv_Flatten(cfg)
    elif cfg.model_name == 'Conv_Pool':
        return Conv_Pool(cfg)
    elif cfg.model_name == 'CustomEEG_Conv':
        return CustomEEG_Conv(cfg)
    elif cfg.model_name == 'Wavelet_Conv':
        return Wavelet_Conv(cfg)
    elif cfg.model_name == 'No_LSTM':
        return No_LSTM(cfg)
    elif cfg.model_name == 'EEG_Attention':
        return EEG_Attention(cfg)
    elif cfg.model_name == 'Ensemble':
        return Ensemble(cfg)

class Ensemble(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.ensemble_type = cfg.ensemble_type
        self.model_list = nn.ModuleList([
            Conv_Flatten(cfg.Conv_Flatten).load_state_dict(torch.load(cfg.Conv_Flatten.model_path)),
            Conv_Pool(cfg.Conv_Pool).load_state_dict(torch.load(cfg.Conv_Pool.model_path))])

        num_model = len(self.model_list)
        self.ensemble_linear = nn.Linear(num_model * 6, 6)

    def forward(self, x):
        
        outs = []
        with torch.no_grad():
            for model in self.model_list:
                out = model(x)
                outs.append(out)
        
        outs = torch.stack(outs)
        outs = outs.permute(1,0,2)
        
        if self.ensemble_type == 'mean':
            ensembled = torch.mean(outs, dim=1)
        elif self.ensemble_type == 'linear':
            outs = torch.flatten(outs, 1)
            ensembled = self.ensemble_linear(outs)
        return ensembled
        
class Conv_Flatten(nn.Module):
    
    def __init__(self, cfg=Config({"in_channel":20, "hidden_channel":32,"out_channel":6,
                                   "kernel_size":5, "pool":[4,4,5,5],"bidirectional":True,
                                   "sequence_length":10000})):  
        
        super().__init__()
        padding = int((cfg.kernel_size - 1) // 2)

        self.lstm1 = nn.LSTM(cfg.in_channel, cfg.hidden_channel, batch_first=True, 
                             bidirectional=cfg.bidirectional)

        self.conv_layer = Expansion(in_channel=cfg.hidden_channel*2, 
                                    kernel_size=cfg.kernel_size, pool=cfg.pool)
        
        self.lstm2 = nn.LSTM(cfg.hidden_channel * 32, cfg.hidden_channel * 32,
                             batch_first=True, bidirectional=cfg.bidirectional)
        
        linear_input_size = cfg.hidden_channel * 32 * int(cfg.sequence_length/math.prod(cfg.pool))
        if cfg.bidirectional == True:
            linear_input_size = linear_input_size * 2
        
        self.last_linear = nn.Sequential(OrderedDict([
            ('linear4', nn.Linear(linear_input_size, cfg.hidden_channel * 32)),
            ('activate4', nn.LeakyReLU()),
            ('linear3', nn.Linear(cfg.hidden_channel * 32, cfg.hidden_channel * 8)),
            ('activate3', nn.LeakyReLU()),
            ('linear2', nn.Linear(cfg.hidden_channel * 8, cfg.hidden_channel * 2)),
            ('activate2', nn.LeakyReLU()),
            ('linear1', nn.Linear(cfg.hidden_channel * 2, cfg.out_channel)),
        ]))
        
    def forward(self, x):
        x = x.permute(0,2,1)
        out, _ = self.lstm1(x)
        out = out.permute(0,2,1)
        
        out = self.conv_layer(out)
        out = out.permute(0,2,1)
        out, _ = self.lstm2(out)
        out = torch.flatten(out, 1).squeeze()
        out = self.last_linear(out)
        
        return out
    
class Conv_Pool(nn.Module):
    
    def __init__(self, cfg=Config({"in_channel":20, "hidden_channel":32,"out_channel":6,
                                   "kernel_size":5, "pool":[4,4,5,5],"bidirectional":True})):  
        
        super().__init__()
        
        padding = int((cfg.kernel_size - 1) // 2)
        self.lstm1 = nn.LSTM(cfg.in_channel, cfg. hidden_channel, 
                              batch_first=True, bidirectional=cfg.bidirectional)
        
        self.conv_layer = Expansion(in_channel=cfg.hidden_channel*2, kernel_size=cfg.kernel_size, pool=cfg.pool)
        
        self.lstm2 = nn.LSTM(cfg.hidden_channel * 32, cfg.hidden_channel * 32,
                             batch_first=True, bidirectional=cfg.bidirectional)
        self.last_linear = nn.Sequential(OrderedDict([
            ('linear4', nn.Linear(cfg.hidden_channel * 64, cfg.hidden_channel * 32)),
            ('activate4', nn.LeakyReLU()),
            ('linear3', nn.Linear(cfg.hidden_channel * 32, cfg.hidden_channel * 8)),
            ('activate3', nn.LeakyReLU()),
            ('linear2', nn.Linear(cfg.hidden_channel * 8, cfg.hidden_channel * 2)),
            ('activate2', nn.LeakyReLU()),
            ('linear1', nn.Linear(cfg.hidden_channel * 2, cfg.out_channel)),
        ]))
        
    def forward(self, x):
        x = x.permute(0,2,1)
        out, _ = self.lstm1(x)
        out = out.permute(0,2,1)
        out = self.conv_layer(out)
        out = out.permute(0,2,1)
        out, _ = self.lstm2(out)
        out = out.permute(0,2,1)
        out = F.avg_pool1d(out, out.shape[-1]).squeeze()
        out = self.last_linear(out)
        
        return out
    
class No_LSTM(nn.Module):
    
    def __init__(self, cfg=Config({"in_channel":20, "hidden_channel":32,"out_channel":6,
                                   "kernel_size":[5,5,5,5,5], "pool":[2,2,4,5,5], "alpha":2})):  
        
        super().__init__()
        
        padding = int((cfg.kernel_size[0] - 1) // 2)
        self.first_conv = nn.Conv1d(cfg.in_channel, cfg.hidden_channel, 
                                    kernel_size=cfg.kernel_size[0], padding=padding)
        
        self.conv_layer = Expansion(in_channel=cfg.hidden_channel, kernel_size=cfg.kernel_size, pool=cfg.pool, alpha=cfg.alpha)
        expand_size = int(cfg.hidden_channel * (cfg.alpha**len(cfg.pool)))
        self.last_linear = nn.Sequential(OrderedDict([
            ('linear4', nn.Linear(expand_size, expand_size // 2)),
            ('activate4', nn.LeakyReLU()),
            ('linear3', nn.Linear(expand_size // 2, expand_size // 4)),
            ('activate3', nn.LeakyReLU()),
            ('linear2', nn.Linear(expand_size // 4, expand_size // 8)),
            ('activate2', nn.LeakyReLU()),
            ('linear1', nn.Linear(expand_size // 8, cfg.out_channel)),
        ]))
        
    def forward(self, x):
        out = self.first_conv(x)
        out = self.conv_layer(out)
        out = F.avg_pool1d(out, out.shape[-1]).squeeze()
        out = self.last_linear(out)
        
        return out

class EEG_Attention(nn.Module):
    
    def __init__(self, cfg=Config({"in_channel":20, "hidden_channel":32,"out_channel":6,
                                   "kernel_size":[5,5,5,5,5], "pool":[2,2,4,5,5], "alpha":2,
                                   "attention_kernel_size":[5,5]})):  
        
        super().__init__()
        
        padding = int((cfg.kernel_size[0] - 1) // 2)
        
        self.attention = Waveform_Attention(cfg.in_channel, cfg.attention_kernel_size)
        self.first_conv = nn.Conv1d(cfg.in_channel, cfg.hidden_channel, 
                                    kernel_size=cfg.kernel_size[0], padding=padding)
        
        self.conv_layer = Expansion(in_channel=cfg.hidden_channel, kernel_size=cfg.kernel_size, pool=cfg.pool, alpha=cfg.alpha)
        expand_size = int(cfg.hidden_channel * (cfg.alpha**len(cfg.pool)))
        self.last_linear = nn.Sequential(OrderedDict([
            ('linear4', nn.Linear(expand_size, expand_size // 2)),
            ('activate4', nn.LeakyReLU()),
            ('linear3', nn.Linear(expand_size // 2, expand_size // 4)),
            ('activate3', nn.LeakyReLU()),
            ('linear2', nn.Linear(expand_size // 4, expand_size // 8)),
            ('activate2', nn.LeakyReLU()),
            ('linear1', nn.Linear(expand_size // 8, cfg.out_channel)),
        ]))
        
    def forward(self, x):
        out = self.attention(x)
        out = self.first_conv(x)
        out = self.conv_layer(out)
        out = F.avg_pool1d(out, out.shape[-1]).squeeze()
        out = self.last_linear(out)
        
        return out
    
class CustomEEG_Conv(nn.Module):
    def __init__(self, cfg=Config({"in_channels":[4,4,4,4], "out_channel":6,
                                   "kernel_size":[5,5,5,5,5], "pool":[2,2,4,5,5], "alpha":2})):
        super().__init__()
        self.n_splits = len(cfg.in_channels)
        
        self.each_conv = nn.ModuleList([
            Expansion(in_channel, kernel_size=cfg.kernel_size, pool=cfg.pool, alpha=cfg.alpha)
            for in_channel in cfg.in_channels
        ])
        
        linear_input_size =  sum([in_channel * cfg.alpha ** len(cfg.pool) for in_channel in cfg.in_channels])
            
        self.last_linear = nn.Sequential(OrderedDict([
            ('linear6', nn.Linear(linear_input_size, linear_input_size//2)),
            ('linear5', nn.Linear(linear_input_size//2, linear_input_size//4)),
            ('linear4', nn.Linear(linear_input_size//4, linear_input_size//8)),
            ('linear3', nn.Linear(linear_input_size//8, linear_input_size//16)),
            ('linear2', nn.Linear(linear_input_size//16,linear_input_size//32)),
            ('linear1', nn.Linear(linear_input_size//32, cfg.out_channel)),
        ]))        
        
        
    def forward(self, x):
        x = x.permute(1,0,2,3)
        outs = []
        for i, _x in enumerate(x):
            _out = self.each_conv[i](_x)
            _out = F.avg_pool1d(_out, _out.shape[-1]).squeeze(dim=2)
            outs.append(_out)
        out = torch.stack(outs).permute(1,0,2)
        out = torch.flatten(out, 1)
        out = self.last_linear(out)
        return out
    
class Wavelet_Conv(nn.Module):
    
    def __init__(self, cfg=Config({"in_channel":4, "num_bin":30, "hidden_channel":32,"out_channel":6,
                                   "kernel_size":[5,5,5,5,5], "pool_channel":[4,4,5,5,5], "alpha":2})):  
        
        super().__init__()

        self.expand = Expansion_2d(in_feature=cfg.in_channel,in_channel=cfg.num_bin,
                                   kernel_size=cfg.kernel_size, pool_channel=cfg.pool_channel,
                                   alpha=cfg.alpha)

        expand_size = cfg.in_channel * cfg.num_bin * int(cfg.alpha**len(cfg.pool_channel))
        
        self.last_linear = nn.Sequential(OrderedDict([
            ('linear4', nn.Linear(expand_size, expand_size // 2)),
            ('activate4', nn.LeakyReLU()),
            ('linear3', nn.Linear(expand_size // 2, expand_size // 4)),
            ('activate3', nn.LeakyReLU()),
            ('linear2', nn.Linear(expand_size // 4, expand_size // 8)),
            ('activate2', nn.LeakyReLU()),
            ('linear1', nn.Linear(expand_size // 8, cfg.out_channel)),
        ]))
        
    def forward(self, x):
        
        out = self.expand(x)
        out = torch.flatten(out, 1)
        out = self.last_linear(out)
        
        return out
    
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg : DictConfig):
    x = torch.rand(32,4,4,10000)
    model = CustomEEG_Conv()
    out = model(x)
    print(out.shape)
     
if __name__ == "__main__":
    main()