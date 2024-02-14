import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold, train_test_split

from dataset import BufferDataset, NonOverlapDataset, collate_fn_nonoverlap
from models import Conv_LSTM, get_model
from loss import KLDivLoss
from analysis import AccuracyTable

import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from collections import OrderedDict
from pathlib import Path
import numpy as np
import time
import datetime
import os
from logging import getLogger ,StreamHandler, FileHandler, Formatter, INFO

device = torch.device('cuda')

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg : DictConfig, debug=False):
    if debug:
        torch.autograd.set_detect_anomaly(True)

    # working directory------------------------------------------------------------------------------
    cwd                 = Path(get_original_cwd())
    checkpoint_dir      = cwd / "output" / cfg.name / "checkpoint"
    config_dir          = cwd / "output" / cfg.name / "config"
    tensorboard_dir     = cwd / "output" / cfg.name / "tensorboard"
    logging_dir         = cwd / "output" / cfg.name / "log"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)
    
    # set logger-------------------------------------------------------------------------------------
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    format = "%(asctime)s [%(filename)s:%(lineno)d] %(message)s"
    fl_handler = FileHandler(filename=(logging_dir/"train.log"), mode='w',encoding="utf-8")
    fl_handler.setFormatter(Formatter(format))
    fl_handler.setLevel(INFO)
    logger.addHandler(fl_handler)
    logger.info("train start")
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("---------------------------------------------------------------------------\n")
    
    # define dataset / dataloader -------------------------------------------------------------------
    dataset     = NonOverlapDataset(cfg.dir.input)
    train_index, valid_index = train_test_split(range(len(dataset)),
                                                test_size=cfg.train.test_size,random_state=0)
    trainset = Subset(dataset, train_index)
    validset = Subset(dataset, valid_index)
    
    trainloader  = torch.utils.data.DataLoader(trainset, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=collate_fn_nonoverlap)
    validloader  = torch.utils.data.DataLoader(validset, batch_size=cfg.train.batch_size, shuffle=False, collate_fn=collate_fn_nonoverlap)
    
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
    writer          = SummaryWriter(log_dir=tensorboard_dir)
    current_step    = 1
    times           = np.array([])
    acc_table       = AccuracyTable()
    
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
                logger.info(msg)
                
            if current_step % cfg.step.save == 0:
                
                path = checkpoint_dir / "{}.pth".format(current_step)
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
                    logger.info(msg)
                    writer.add_scalar("valid loss", loss, current_step)
            
            # batch end---------------------------------------------------------------        
            end             = time.perf_counter()
            times = np.append(times, end - start)
            current_step += 1
            
        #epoch end---------------------------------------------------------------------
        scheduler.step()
        
if __name__ == "__main__":
    main()