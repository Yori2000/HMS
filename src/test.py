import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataset import HmsTrainDataset2
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
import time
import datetime
import os

device = torch.device('cuda')

class SimpleModel(nn.Module):
    
    def __init__(self, in_channel=20, out_channel=6):  
        
        super().__init__()
        
        self.lstm1 = nn.LSTM(in_channel, 32, batch_first=True, bidirectional=True)
        
        self.conv_layer = nn.Sequential(OrderedDict([
            ('conv1',       nn.Conv1d(64, 32, kernel_size=5, padding=2)),
            ('batch_norm1', nn.BatchNorm1d(32)),
            ('relu1',       nn.ReLU(32)),
            ('pooling1',    nn.AvgPool1d(4)),
            
            ('conv2',       nn.Conv1d(32, 32, kernel_size=5, padding=2)),
            ('batch_norm2', nn.BatchNorm1d(32)),
            ('relu2',       nn.ReLU(32)),
            ('pooling2',    nn.AvgPool1d(5)),  
            
            ('conv3',       nn.Conv1d(32, 32, kernel_size=5, padding=2)),
            ('batch_norm3', nn.BatchNorm1d(32)),
            ('relu3',       nn.ReLU(32)),
            ('pooling3',    nn.AvgPool1d(5)),   
        ]))
        
        self.lstm2 = nn.LSTM(32, 32, batch_first=True, bidirectional=True)
        
        self.last_linear = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(64 * 100, 512)),
            ('linear2', nn.Linear(512, 256)),
            ('linear3', nn.Linear(256, 64)),
            ('linear4', nn.Linear(64, out_channel))
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
    
class SimpleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_vote      = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, y, vote_target, consensus_target):
        
        vote_probability    = F.normalize(vote_target.to(torch.float32), dim=1)
        vote_loss           = self.loss_vote(torch.log(y), vote_probability)
        
        return vote_loss

def train():
    
    test_name = "double_lstm"
    num_epochs = 200
    batch_size = 32
    learning_rate = 1e-4
  
    dataset     = HmsTrainDataset2("./data", "./data/train.csv")
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # validloader  = torch.utils.data.DataLoader(validset, batch_size=32, shuffle=False)
    
    model       = SimpleModel().to(device)
    loss_fn     = SimpleLoss()
    optimizer   = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    scheduler   = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

    total_step      = int((len(dataset) / batch_size) * num_epochs) 
    checkpoint_dir  = Path("./checkpoint/{}".format(test_name))
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_step        = 50
    checkpoint_step = 25000
    valid_step      = 20
    writer          = SummaryWriter(log_dir=(checkpoint_dir / "logs"))
    
    print("train start at : {}\ntotal_step : {}, step per epoch : {}\nepoch : {}, batch size : {}, learning rate : {}"
          .format(datetime.datetime.now(), total_step, int(len(dataset)/batch_size), num_epochs, batch_size, learning_rate))
    
    model.train()
    current_step    = 1
    times           = np.array([])
    for epoch in range(num_epochs):
        for batch in trainloader:
            start = time.perf_counter()
            
            eeg         = batch[0].to(device)
            consensus   = batch[1].to(device)
            vote        = batch[2].to(device)

            out     = model(eeg)
            loss    = loss_fn(out, vote, consensus)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_lr = scheduler.get_last_lr()[0]
            
            writer.add_scalar("train loss", loss.item(), current_step)
            writer.add_scalar("learning rate", current_lr, current_step)
            
            if current_step % log_step == 0:
                msg     = "Epoch : {}, Step : {}({:.3f}%), Loss : {:.5f}, Remaining Time : {:.3f}"\
                          .format(epoch, current_step, (current_step / total_step)*100, loss.item(),
                                    (total_step - current_step)*np.mean(times))
                print(msg)
                
            if current_step % checkpoint_step == 0:
                
                path = checkpoint_dir / "{}.pth".format(current_step)
                torch.save(model.state_dict(), path)
            
            # if current_step % valid_step == 0:
            #     with torch.no_grad():
            #         for b in validloader:
            #             eeg         = batch[0].to(device)
            #             consensus   = batch[1]
            #             vote        = batch[2].to(device)

            #             out              = model(eeg)
            #             vote_probability = F.normalize(vote.to(torch.float32), dim=1)
            #             loss             = F.cross_entropy(out, vote_probability)

            #             msg = "Valid\tEpoch : {}, Step : {}, valid Loss : {}"\
            #                   .format(epoch, current_step, loss.item())
            #             print(msg)
            #             writer.add_scalar("valid loss", loss.item(), current_step)
            end             = time.perf_counter()
            times = np.append(times, end - start)
            current_step += 1
        scheduler.step()
        
if __name__ == "__main__":
    
    train()