import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Subset
from sklearn.model_selection import KFold, train_test_split

from dataset import BufferDataset
from models import Conv_LSTM, get_model
from loss import KLDivLoss
from analysis import AccuracyTable

import hydra
from omegaconf import DictConfig
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import numpy as np
import time
import datetime
import os

device = torch.device('cuda')

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg : DictConfig, debug=False):
    if debug:
        torch.autograd.set_detect_anomaly(True)
    
    # define dataset / dataloader -------------------------------------------------------------------
    dataset     = BufferDataset(cfg.dir.input, idealized=True)
    train_index, valid_index = train_test_split(range(len(dataset)),
                                                test_size=cfg.train.test_size,random_state=0)
    trainset = Subset(dataset, train_index)
    validset = Subset(dataset, valid_index)
    
    trainloader  = torch.utils.data.DataLoader(trainset, batch_size=cfg.train.batch_size, shuffle=True)
    validloader  = torch.utils.data.DataLoader(validset, batch_size=cfg.train.batch_size, shuffle=False)
    
    # define model / loss function / optimizer ------------------------------------------------------
    model       = get_model(cfg.model).to(device)
    loss_fn     = KLDivLoss()
    
    optimizer   = optim.Adam(model.parameters(), lr=cfg.optim.learning_rate, 
                             weight_decay=cfg.optim.weight_decay)
    scheduler   = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_scheduler.step,
                                            gamma=cfg.lr_scheduler.gamma)
    
    # prepare for train---------------------------------------------------------------
    model.train()
    total_step  = int(len(trainset) / cfg.train.batch_size * cfg.train.epochs)
    writer          = SummaryWriter(log_dir=(Path(cfg.dir.save) / "logs"))
    current_step    = 1
    times           = np.array([])
    acc_table       = AccuracyTable()
    print("train start at : {}\ntotal_step : {}, epoch : {}, batch size : {}, learning rate : {}"
          .format(datetime.datetime.now(), total_step, 
                  cfg.train.epochs, cfg.train.batch_size, cfg.optim.learning_rate))
    
    # train start----------------------------------------------------------------------
    for epoch in range(cfg.train.epochs):
        for batch in trainloader:
            start = time.perf_counter()
            
            # calculate prediction----------------------------------------------
            eeg         = batch[0].to(device)
            consensus   = batch[1].to(device)
            vote        = batch[2].to(device)

            out     = model(eeg)
            loss    = loss_fn(out, vote)
            
            # optimeze model------------------------------------------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # analysis / logging / save model -----------------------------------------------------
            writer.add_scalar("train loss", loss.item(), current_step)
            writer.add_scalar("learning rate", current_lr, current_step)
            acc_table.add(out, consensus)
            acc_table.write_tensorboard(writer, current_step)
            
            if current_step % cfg.step.log == 0:
                msg     = "Epoch : {}, Step : {}({:.3f}%), Loss : {:.5f}, Remaining Time : {:.3f}"\
                          .format(epoch, current_step, (current_step / total_step)*100, loss.item(),
                                 (total_step - current_step)*np.mean(times))
                print(msg)
                
            if current_step % cfg.step.save == 0:
                
                path = cfg.dir.save / "{}.pth".format(current_step)
                torch.save(model.state_dict(), path)
            
            # validation ------------------------------------------------------------------------
            if current_step % cfg.step.valid == 0:
                with torch.no_grad():
                    counter = 0
                    valid_loss  = np.array([])
                    for b in validloader:
                        eeg         = b[0].to(device)
                        consensus   = b[1]
                        vote        = b[2].to(device)

                        out              = model(eeg)
                        loss             = loss_fn(out, vote)
                        valid_loss = np.append(valid_loss, loss.clone().to('cpu').detach().numpy())
                    loss = np.mean(valid_loss)
                    msg = "Valid\tEpoch : {}, Step : {}, valid Loss : {}"\
                            .format(epoch, current_step, loss)
                    print(msg)
                    writer.add_scalar("valid loss", loss, current_step)
                        
            end             = time.perf_counter()
            times = np.append(times, end - start)
            current_step += 1
        scheduler.step()
        
if __name__ == "__main__":
    
    main()