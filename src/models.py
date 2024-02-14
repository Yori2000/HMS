import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from modules import ConvLSTM
from utils import Config

from collections import OrderedDict
from omegaconf import DictConfig
from logging import getLogger

logger = getLogger('main').getChild('model')


def get_model(cfg):
    models = []
    if cfg.name =='Conv_LSTM':
        return Conv_LSTM(cfg)
    elif cfg.name =='Double_LSTM':
        return Double_LSTM(cfg)
    elif cfg.name == 'LSTM_Pool':
        return LSTM_Pool(cfg)
    elif cfg.name == 'EEG_Split_Conv':
        return EEG_Split_Conv(cfg)

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
    
    
class Double_LSTM(nn.Module):
    
    def __init__(self, cfg):  
        
        super().__init__()
        
        padding = int((cfg.kernel_size - 1) // 2)
        
        self.lstm1 = nn.LSTM(cfg.in_channel, cfg.hidden_channel, batch_first=True, 
                             bidirectional=cfg.bidirectional)
        self.conv_layer = nn.Sequential(OrderedDict([
            ('conv1',       nn.Conv1d(cfg.hidden_channel*2, cfg.hidden_channel*2, 
                                      kernel_size=cfg.kernel_size, padding=padding)),
            ('batch_norm1', nn.BatchNorm1d(cfg.hidden_channel*2)),
            ('relu1',       nn.ReLU(cfg.hidden_channel*2)),
            ('pooling1',    nn.AvgPool1d(4)),
            
            ('conv2',       nn.Conv1d(cfg.hidden_channel*2, cfg.hidden_channel * 2,
                                      kernel_size=cfg.kernel_size, padding=padding)),
            ('batch_norm2', nn.BatchNorm1d(cfg.hidden_channel * 2)),
            ('relu2',       nn.ReLU(cfg.hidden_channel * 2)),
            ('pooling2',    nn.AvgPool1d(5)),
            
            ('conv3',       nn.Conv1d(cfg.hidden_channel * 2, cfg.hidden_channel * 2,
                                      kernel_size=cfg.kernel_size, padding=padding)),
            ('batch_norm3', nn.BatchNorm1d(cfg.hidden_channel * 2)),
            ('relu3',       nn.ReLU(cfg.hidden_channel * 2)),
            ('pooling3',    nn.AvgPool1d(5))
        ]))
        
        self.lstm2 = nn.LSTM(cfg.hidden_channel * 2, cfg.hidden_channel * 2,
                             batch_first=True, bidirectional=cfg.bidirectional)
        
        linear_input_size = cfg.hidden_channel * 2 * int(cfg.sequence_length/100)
        if cfg.bidirectional == True:
            linear_input_size = linear_input_size * 2
        
        self.last_linear = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(linear_input_size, 512)),
            ('linear2', nn.Linear(512, 256)),
            ('linear3', nn.Linear(256, 64)),
            ('linear4', nn.Linear(64, cfg.out_channel))
        ]))
        
    def forward(self, x):
        x = x.permute(0,2,1)
        out, _ = self.lstm1(x)
        out = out.permute(0,2,1)
        
        out = self.conv_layer(out)
        out = out.permute(0,2,1)
        out, _ = self.lstm2(out)
        out = torch.flatten(out, 1)
        out = self.last_linear(out)
        out = F.softmax(out, dim=1)
        
        return out
    
class LSTM_Pool(nn.Module):
    
    def __init__(self, cfg=Config({"in_channel":20, "hidden_channel":32,"out_channel":6,
                                   "kernel_size":5, "bidirectional":True})):  
        
        super().__init__()
        
        padding = int((cfg.kernel_size - 1) // 2)
        # self.lstm1 = nn.LSTM(cfg.in_channel, cfg. hidden_channel, 
        #                      batch_first=True, bidirectional=cfg.bidirectional)
        
        self.conv_layer = nn.Sequential(OrderedDict([
            ('conv1',       nn.Conv1d(cfg.in_channel, cfg.hidden_channel, 
                                      kernel_size=cfg.kernel_size, padding=padding)),
            ('batch_norm1', nn.BatchNorm1d(cfg.hidden_channel)),
            ('relu1',       nn.ReLU(cfg.hidden_channel)),
            ('pooling1',    nn.AvgPool1d(4)),
            
            ('conv2',       nn.Conv1d(cfg.hidden_channel, cfg.hidden_channel * 2,
                                      kernel_size=cfg.kernel_size, padding=padding)),
            ('batch_norm2', nn.BatchNorm1d(cfg.hidden_channel * 2)),
            ('relu2',       nn.ReLU(cfg.hidden_channel * 2)),
            ('pooling2',    nn.AvgPool1d(4)),
            
            ('conv3',       nn.Conv1d(cfg.hidden_channel * 2, cfg.hidden_channel * 4,
                                      kernel_size=cfg.kernel_size, padding=padding)),
            ('batch_norm3', nn.BatchNorm1d(cfg.hidden_channel * 4)),
            ('relu3',       nn.ReLU(cfg.hidden_channel * 4)),
            ('pooling3',    nn.AvgPool1d(5)),
            
            ('conv4',       nn.Conv1d(cfg.hidden_channel * 4, cfg.hidden_channel * 8,
                                      kernel_size=cfg.kernel_size, padding=padding)),
            ('batch_norm4', nn.BatchNorm1d(cfg.hidden_channel * 8)),
            ('relu4',       nn.ReLU(cfg.hidden_channel * 8)),
            ('pooling4',    nn.AvgPool1d(5))
        ]))
        
        self.lstm2 = nn.LSTM(cfg.hidden_channel * 8, cfg.hidden_channel * 16,
                             batch_first=True, bidirectional=cfg.bidirectional)
        
        self.last_linear = nn.Sequential(OrderedDict([
            ('linear6', nn.Linear(cfg.hidden_channel * 32, cfg.hidden_channel * 16)),
            ('linear5', nn.Linear(cfg.hidden_channel * 16, cfg.hidden_channel * 8)),
            ('linear4', nn.Linear(cfg.hidden_channel * 8, cfg.hidden_channel * 4)),
            ('linear3', nn.Linear(cfg.hidden_channel * 4, cfg.hidden_channel*2)),
            ('linear2', nn.Linear(cfg.hidden_channel * 2, cfg.hidden_channel)),
            ('linear1', nn.Linear(cfg.hidden_channel, cfg.out_channel)),
        ]))
        
    def forward(self, x):
        # x = x.permute(0,2,1)
        # out, _ = self.lstm1(x)
        # out = out.permute(0,2,1)
        out = self.conv_layer(x)
        out = out.permute(0,2,1)
        out, _ = self.lstm2(out)
        out = out.permute(0,2,1)
        out = F.avg_pool1d(out, out.shape[-1]).squeeze()
        out = self.last_linear(out)
        out = F.softmax(out, dim=0)
        
        return out
    
class EEG_Split_Conv(nn.Module):
    def __init__(self, cfg=Config({"in_channels":[5,5,5,5,4], "hidden_channel":8,"out_channel":6,
                                   "kernel_size":5, "bidirectional":True})):
        super().__init__()
        self.n_splits = len(cfg.in_channels)
        self.LL_index = {"Fp1":0, "F7":4, "T3":5, "T5":6, "O1":7}
        self.LP_index = {"Fp1":0, "F3":1, "C3":2, "P3":3, "O1":7}
        self.RL_index = {"Fp2":11, "F8":15, "T4":16, "T6":17, "O2":18}
        self.RP_index = {"Fp2":11, "F4":12, "C4":13, "P4":14, "O2":18}
        self.Other_index = {"Fz":8,"Cz":9,"Pz":10,"EKG":19}
        
        self.each_conv = nn.ModuleList([
            ConvLSTM(input_channel=i, hidden_channel=cfg.hidden_channel,
                     kernel_size=cfg.kernel_size, bidirectional=cfg.bidirectional)
            for i in cfg.in_channels
        ])
        
        conv_out_size = cfg.hidden_channel * 8 * 2 * len(cfg.in_channels)
        self.lstm = nn.LSTM(conv_out_size, conv_out_size,
                             batch_first=True, bidirectional=cfg.bidirectional)
        
        linear_input_size = conv_out_size
        if cfg.bidirectional:
            linear_input_size = linear_input_size * 2
            
        self.last_linear = nn.Sequential(OrderedDict([
            ('linear6', nn.Linear(linear_input_size, linear_input_size//2)),
            ('linear5', nn.Linear(linear_input_size//2, linear_input_size//4)),
            ('linear4', nn.Linear(linear_input_size//4, linear_input_size//8)),
            ('linear3', nn.Linear(linear_input_size//8, linear_input_size//16)),
            ('linear2', nn.Linear(linear_input_size//16,linear_input_size//32)),
            ('linear1', nn.Linear(linear_input_size//32, cfg.out_channel)),
        ]))        
        
        
    def forward(self, x):
        LL = torch.stack([x[:,v,:] for v in self.LL_index.values()]).permute(1,0,2)
        LP = torch.stack([x[:,v,:] for v in self.LP_index.values()]).permute(1,0,2)
        RL = torch.stack([x[:,v,:] for v in self.RL_index.values()]).permute(1,0,2)
        RP = torch.stack([x[:,v,:] for v in self.RP_index.values()]).permute(1,0,2)
        Other = torch.stack([x[:,v,:] for v in self.Other_index.values()]).permute(1,0,2)
        inputs = [LL, LP, RL, RP, Other]
        
        outs = []
        for i, _x in enumerate(inputs):
            _out = self.each_conv[i](_x)
            outs.append(_out)
        out = torch.cat((outs[0],outs[1],outs[2],outs[3],outs[4]), dim=2)

        out, _ = self.lstm(out)
        out = out.permute(0,2,1)
        out = F.avg_pool1d(out, out.shape[-1]).squeeze()
        out = self.last_linear(out)
        out = F.softmax(out, dim=0)
        return out
    
if __name__ == "__main__":
    
    x = torch.rand(32,20,10000)
    model = EEG_Split_Conv()
    out = model(x)
    print(out.shape)