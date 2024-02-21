import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Subset

from sklearn.model_selection import KFold, train_test_split

from dataset import get_dataset
from models import get_model
from loss import KLDivLoss
from analysis import AccuracyTable
from utils import setup

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import numpy as np
import time
import wandb

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg : DictConfig):
    
    cfg, device, logger = setup(cfg)
    
    logger.info("train start")
    logger.info("---------------------------------------------------------------------------\n")
    
    # define dataset / dataloader -------------------------------------------------------------------
    dataset, collate_fn         = get_dataset(cfg.dir.input, cfg.dataset)
    train_index, valid_index    = train_test_split(range(len(dataset)),
                                                test_size=cfg.train.test_size,random_state=0)
    trainset                    = Subset(dataset, train_index)
    validset                    = Subset(dataset, valid_index)
    
    trainloader                 = torch.utils.data.DataLoader(trainset, batch_size=cfg.train.batch_size,
                                                              shuffle=True, collate_fn=collate_fn)
    validloader                 = torch.utils.data.DataLoader(validset, batch_size=cfg.train.batch_size,
                                                              shuffle=False, collate_fn=collate_fn)
    
    # define model / loss function / optimizer ------------------------------------------------------
    model       = get_model(cfg.model).to(device)
    loss_fn     = KLDivLoss()
    
    optimizer   = optim.Adam(model.parameters(), lr=cfg.optim.learning_rate, 
                             weight_decay=cfg.optim.weight_decay)
    scheduler   = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_scheduler.step,
                                            gamma=cfg.lr_scheduler.gamma)
    
    # train start----------------------------------------------------------------------
    try:
        train(cfg, logger, model, loss_fn, optimizer, scheduler, trainloader, validloader, device)
    except Exception as e:
        logger.exception(e)
        

def train(cfg, logger, model, loss_fn, optimizer, scheduler, trainloader, validloader, device):  
    # prepare for train---------------------------------------------------------------
    if cfg.analysis:
        wandb.init(project="HMS", name=cfg.exp_name,config={k: v for k, v in cfg.items() if k!='dir'})
        wandb.watch(model)
    if cfg.debug:
        logger.info("DEBUG")
        torch.autograd.set_detect_anomaly(True)
    model.train()
    total_step      = int(len(trainloader) * cfg.train.epochs)    
    current_step    = 1
    times           = np.array([])
    acc_table       = AccuracyTable()
    # train start----------------------------------------------------------------------
    for epoch in range(cfg.train.epochs):
        for batch in trainloader:
            start = time.perf_counter()
            
            # calculate prediction----------------------------------------------
            x           = batch[0].to(device)
            target      = batch[1].to(device)
            
            out         = model(x)
            loss        = loss_fn(out, target)
            
            logger.debug("out : {}".format(out.detach()))
            logger.debug("Is out include NaN : {}".format(torch.isnan(out)))
            logger.debug("Is loss include NaN : {}".format(torch.isnan(loss)))
            
            # optimeze model------------------------------------------------------------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # logging -----------------------------------------------------
            if current_step % cfg.step.log == 0:
                msg     = "Epoch : {}, Step : {}({:.3f}%), Loss : {:.5f}, Remaining Time : {:.3f}"\
                          .format(epoch, current_step, (current_step / total_step)*100, loss.item(),
                                 (total_step - current_step)*np.mean(times))
                logger.info(msg)
                
                if cfg.analysis:
                    wandb.log({"train_loss": loss.item(), "learning_rate":current_lr}, step=current_step)
            
            # save model -----------------------------------------------------  
            if current_step % cfg.step.save == 0:
                
                path = Path(cfg.dir.checkpoint) / "{}.pth".format(current_step)
                torch.save(model.state_dict(), path)
            
            # validation ------------------------------------------------------------------------
            if current_step % cfg.step.valid == 0:
                with torch.no_grad():
                    counter = 0
                    valid_loss  = np.array([])
                    for b in validloader:
                        x           = b[0].to(device)
                        target      = b[1].to(device)

                        out         = model(x)
                        loss        = loss_fn(out, target)
                        valid_loss = np.append(valid_loss, loss.clone().to('cpu').detach().numpy())
                        acc_table.add(out, torch.argmax(target, dim=1))
                        
                    loss = np.mean(valid_loss)
                    msg = "Valid\tEpoch : {}, Step : {}, valid Loss : {}"\
                            .format(epoch, current_step, loss)
                    logger.info(msg)

                    if cfg.analysis:
                        acc_table.log_wandb(current_step)
                        wandb.log({"valid loss":loss}, step=current_step)
            # batch end---------------------------------------------------------------        
            end             = time.perf_counter()
            times = np.append(times, end - start)
            current_step += 1
            
        #epoch end---------------------------------------------------------------------
        scheduler.step()
        logger.info("epoch {} end".format(epoch))
        logger.info("---------------------------------------------------------------------------\n")
        
    if cfg.analysis:
        wandb.finish()
        
if __name__ == "__main__":
    main()