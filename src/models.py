from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
# from transformers import HubertModel


# EEGNet model
class EEGNet(nn.Module):
    "input shape: (batch_size, 20, 10000)"
    def __init__(self, in_channel=20, output_dim=6):
        super(EEGNet, self).__init__()
        self.ReLU = nn.LeakyReLU()
        self.conv_kernel_size = (11, 5, 5, 5, 5, 5, 5)
        self.conv_channels = (32, 32, 32, 32, 32, 32, 32)
        self.conv_stride = (5, 2, 2, 2, 2, 2, 2)
        self.conv_padding = (5, 2, 2, 2, 2, 2, 2)
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channel, self.conv_channels[0], self.conv_kernel_size[0], self.conv_stride[0]),
            nn.BatchNorm1d(self.conv_channels[0], False),
            nn.LeakyReLU(),
            nn.Conv1d(self.conv_channels[0], self.conv_channels[1], self.conv_kernel_size[1], self.conv_stride[1]),
            nn.BatchNorm1d(self.conv_channels[1], False),
            nn.LeakyReLU(),
            nn.Conv1d(self.conv_channels[1], self.conv_channels[2], self.conv_kernel_size[2], self.conv_stride[2]),
            nn.BatchNorm1d(self.conv_channels[2], False),
            nn.LeakyReLU(),
            nn.Conv1d(self.conv_channels[2], self.conv_channels[3], self.conv_kernel_size[3], self.conv_stride[3]),
            nn.BatchNorm1d(self.conv_channels[3], False),
            nn.LeakyReLU(),
            nn.Conv1d(self.conv_channels[3], self.conv_channels[4], self.conv_kernel_size[4], self.conv_stride[4]),
            nn.BatchNorm1d(self.conv_channels[4], False),
            nn.LeakyReLU(),
            nn.Conv1d(self.conv_channels[4], self.conv_channels[5], self.conv_kernel_size[5], self.conv_stride[5]),
            nn.BatchNorm1d(self.conv_channels[5], False),
            nn.LeakyReLU(),
            nn.Conv1d(self.conv_channels[5], self.conv_channels[6], self.conv_kernel_size[6], self.conv_stride[6]),
            nn.BatchNorm1d(self.conv_channels[6], False)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(896, 128),
            nn.LeakyReLU(),
            nn.Linear(128, output_dim),
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        
        x = self.fc(x)
        # x = F.softmax(x, dim=1)
        return x
    
    def get_model_name(self):
        return "EEGNet"

# 双方向LSTMを使ったEEGNet
class EEGNet2(nn.Module):
    def __init__(self, feature_size=20, hidden_dim=1, output_dim=6):
        super(EEGNet2, self).__init__()
        self.ReLU = nn.LeakyReLU()
        self.feature_size = feature_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(feature_size, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 20000, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )
        
    def forward(self, x):
        # 転置
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # x = F.softmax(x, dim=1)
        return x
    
    def get_model_name(self):
        return "EEGNet2"

    
    
class SimpleModel(nn.Module):
    
    def __init__(self, in_channel=20, out_channel=6):  
        
        super().__init__()
        
        self.conv_layer = nn.Sequential(OrderedDict([
            ('conv1',       nn.Conv1d(in_channel, 32, kernel_size=9, stride=4, padding=4)),
            ('batch_norm1', nn.BatchNorm1d(32)),
            ('relu1',       nn.LeakyReLU(32)),
            # ('pooling1',    nn.MaxPool1d(4)),
            
            ('conv2',       nn.Conv1d(32, 32, kernel_size=11, stride=5, padding=5)),
            ('batch_norm2', nn.BatchNorm1d(32)),
            ('relu2',       nn.LeakyReLU(32)),
            # ('pooling2',    nn.AvgPool1d(5)),  
            
            ('conv3',       nn.Conv1d(32, 32, kernel_size=11, stride=5, padding=5)),
            ('batch_norm3', nn.BatchNorm1d(32)),
            ('relu3',       nn.LeakyReLU(32)),
            # ('pooling3',    nn.AvgPool1d(5)),   
        ]))
        
        self.lstm = nn.LSTM(32, 32, batch_first=True, bidirectional=True)
        
        self.last_linear = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(64 * 100, 512)),
            ('relu1', nn.LeakyReLU()),
            ('linear2', nn.Linear(512, 256)),
            ('relu2', nn.LeakyReLU()),
            ('linear3', nn.Linear(256, 64)),
            ('relu3', nn.LeakyReLU()),
            ('linear4', nn.Linear(64, out_channel))
        ]))
        
    def forward(self, x):
        
        out = self.conv_layer(x)
        out = out.permute(0,2,1)
        out, _ = self.lstm(out)
        out = torch.flatten(out, 1)
        out = self.last_linear(out)
        # out = F.softmax(out, dim=1)
        
        return out
    
    def get_model_name(self):
        return "SimpleModel"