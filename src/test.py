import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataset import HmsTrainDataset
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

import time
import datetime

device = torch.device('cuda')

class SimpleModel(nn.Module):
    
    def __init__(self, in_channel=20, out_channel=6):  
        
        super().__init__()
        
        self.conv_layer = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(in_channel, 32, kernel_size=5)),
            ('batch_norm1', nn.BatchNorm1d(32)),
            ('relu1', nn.ReLU(32)),
            ('pooling1', nn.AvgPool1d(4)),
            
            ('conv2', nn.Conv1d(32, 64, kernel_size=5)),
            ('batch_norm2', nn.BatchNorm1d(64)),
            ('relu2', nn.ReLU(64)),
            ('pooling2', nn.AvgPool1d(4)),  
            
            ('conv3', nn.Conv1d(64, 64, kernel_size=5)),
            ('batch_norm3', nn.BatchNorm1d(64)),
            ('relu3', nn.ReLU(64)),
            ('pooling3', nn.AvgPool1d(4)),   
        ]))
        
        self.linear = nn.Linear(64, out_channel)
        
    def forward(self, x):
        
        out = self.conv_layer(x)
        
        out = F.avg_pool1d(out, out.shape[2]).squeeze()
        out = self.linear(out)
        out = F.softmax(out, dim=1)
        
        return out
    
class SimpleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_vote      = nn.CrossEntropyLoss()
        self.loss_consensus = nn.CrossEntropyLoss()
        
    def forward(self, y, vote_target, consensus_target):
        
        vote_probability    = F.normalize(vote_target.to(torch.float32), dim=1)
        vote_loss           = self.loss_vote(y, vote_probability)
        
        consensus           = torch.eye(6, device=device)[torch.argmax(y, dim=1)]
        consensus_target    = F.one_hot(consensus_target, num_classes=6).to(torch.float32)
        consensus_loss      = self.loss_consensus(consensus, consensus_target)
        
        return vote_loss + consensus_loss

def train():
    
    num_epochs = 200
    batch_size = 32
  
    dataset     = HmsTrainDataset("./data", "./data/train.csv")
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # validloader  = torch.utils.data.DataLoader(validset, batch_size=32, shuffle=False)
    
    model       = SimpleModel().to(device)
    loss_fn     = SimpleLoss()
    optimizer   = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler   = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    total_step      = len(dataset) / batch_size * num_epochs   
    checkpoint_dir  = Path("./checkpoint")
    log_step        = 20
    checkpoint_step = 10000
    valid_step      = 20
    writer          = SummaryWriter(log_dir="./checkpoint/logs")
    
    print("train start at : {}\ntotal_step : {}".format(datetime.datetime.now(), total_step))
    
    current_step    = 1
    times           = []
    for epoch in range(num_epochs):
        for batch in trainloader:
            start = time.time()
            
            eeg         = batch[0].to(device)
            consensus   = batch[1].to(device)
            vote        = batch[2].to(device)

            out     = model(eeg)
            loss    = loss_fn(out, vote, consensus)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            end             = time.time()
            time_per_batch  = end - start
            times.append(time_per_batch)
            
            if current_step % log_step == 0:
                
                remain  = sum(times)/ current_step * (total_step - current_step)
                msg     = "Epoch : {}, Step : {}({:.3f}%), Loss : {:.5f}, Remaining Time : {:.3f}"\
                          .format(epoch, current_step, (current_step / total_step)*100, loss.item(), remain)
                print(msg)
                writer.add_scalar("train_loss", loss.item(), current_step)
                
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
            #             writer.add_scalar("valid_loss", loss.item(), current_step)
            
            current_step += 1

        
if __name__ == "__main__":
    
    train()