import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import OrderedDict
from omegaconf import DictConfig

def get_model(cfg):
    if cfg.name =='Conv_LSTM':
        return Conv_LSTM(cfg)

class Conv_LSTM(nn.Module):
    
    def __init__(self, cfg):  
        
        super().__init__()
        
        padding = int((cfg.kernel_size - 1) // 2)

        self.conv_layer = nn.Sequential(OrderedDict([
            ('conv1',       nn.Conv1d(cfg.in_channel, cfg.hidden_channel, 
                                      kernel_size=cfg.kernel_size, padding=padding)),
            ('batch_norm1', nn.BatchNorm1d(cfg.hidden_channel)),
            ('relu1',       nn.ReLU(cfg.hidden_channel)),
            
            ('conv2',       nn.Conv1d(cfg.hidden_channel, cfg.hidden_channel * 2,
                                      kernel_size=cfg.kernel_size, padding=padding)),
            ('batch_norm2', nn.BatchNorm1d(cfg.hidden_channel * 2)),
            ('relu2',       nn.ReLU(cfg.hidden_channel * 2)),
            
            ('conv3',       nn.Conv1d(cfg.hidden_channel * 2, cfg.hidden_channel * 2,
                                      kernel_size=cfg.kernel_size, padding=padding)),
            ('batch_norm3', nn.BatchNorm1d(cfg.hidden_channel * 2)),
            ('relu3',       nn.ReLU(cfg.hidden_channel * 2)),
        ]))
        
        self.lstm2 = nn.LSTM(cfg.hidden_channel * 2, cfg.hidden_channel * 2,
                             batch_first=True, bidirectional=cfg.bidirectional)
        
        linear_input_size = cfg.hidden_channel * 2 * cfg.sequence_length
        if cfg.bidirectional == True:
            linear_input_size = linear_input_size * 2
        
        self.last_linear = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(linear_input_size, 512)),
            ('linear2', nn.Linear(512, 256)),
            ('linear3', nn.Linear(256, 64)),
            ('linear4', nn.Linear(64, cfg.out_channel))
        ]))
        
    def forward(self, x):
        
        out = self.conv_layer(x)
        out = out.permute(0,2,1)
        out, _ = self.lstm2(out)
        out = torch.flatten(out, 1)
        out = self.last_linear(out)
        out = F.softmax(out, dim=1)
        
        return out