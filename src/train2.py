import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

from dataset import get_dataset
from models import get_model
from loss import KLDivLoss
from analysis import AccuracyTable
from utils import setup, create_df

import hydra
import polars as pl
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import numpy as np
import time
import wandb

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg : DictConfig):
    
    cfg, device, logger = setup(cfg)
    
    logger.info("process start")
    logger.info("---------------------------------------------------------------------------\n")
    
    # train start----------------------------------------------------------------------
    try:
        train(cfg, logger, device)
    except Exception as e:
        logger.exception(e)
        

def train(cfg, logger, device): 
    
    # define df -------------------------------------------------------------------
    csv = pl.read_csv(Path(cfg.dir.input) / 'train.csv')
    df = create_df(csv, cfg.train.folds)

    # train start----------------------------------------------------------------------
    if cfg.analysis:
        wandb.init(project="HMS", name=cfg.exp_name,config={k: v for k, v in cfg.items() if k!='dir'})

    if cfg.debug:
        logger.info("DEBUG")
        torch.autograd.set_detect_anomaly(True)
        
    for fold in range(cfg.train.folds):
        # define model / loss function / optimizer ------------------------------------------------------
        model       = get_model(cfg.model).to(device)
        loss_fn     = KLDivLoss()  
        
        trainbuf = df.filter(pl.col('fold') != fold)
        validbuf = df.filter(pl.col('fold') == fold)
        
        trainset, collate_fn = get_dataset(cfg.dataset.dataset_name, trainbuf, cfg.dir.input, cfg.dataset.target)
        validset, _          = get_dataset(cfg.dataset.dataset_name, validbuf, cfg.dir.input, cfg.dataset.target)
        
        trainloader = DataLoader(trainset,batch_size=cfg.train.batch_size,shuffle=True,
                                 num_workers=cfg.train.num_workers, pin_memory=True, drop_last=True,
                                 collate_fn=collate_fn)
        validloader = DataLoader(validset,batch_size=cfg.train.batch_size, shuffle=True,
                                 num_workers=cfg.train.num_workers, pin_memory=True, drop_last=True,
                                 collate_fn=collate_fn)
        
        optimizer   = optim.AdamW(model.parameters(), lr=cfg.optim.learning_rate, 
                                  weight_decay=cfg.optim.weight_decay)
        scheduler = OneCycleLR(optimizer,max_lr=1e-3,epochs=cfg.train.epochs,
                               steps_per_epoch=len(trainloader),pct_start=0.1,anneal_strategy="cos",
                               final_div_factor=100)
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.amp)
        
        model.train()
        total_step      = int(len(trainloader) * cfg.train.epochs)    
        current_step    = 1
        times           = np.array([])
        best_loss       = np.inf
        
        # prepare for train---------------------------------------------------------------        
        for epoch in range(cfg.train.epochs):
            for batch in trainloader:
                start = time.perf_counter()
                
                # calculate prediction----------------------------------------------
                x           = batch[0].to(device)
                target      = batch[1].to(device)
                
                with torch.cuda.amp.autocast(enabled=cfg.train.amp):
                    out         = model(x)
                    logger.debug("x : {}, target : {}, out : {}".format(x.shape, target.shape, out.shape))
                    loss        = loss_fn(out, target)

                # optimeze model------------------------------------------------------------
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                
                # logging -----------------------------------------------------
                if current_step % cfg.step.log == 0:
                    msg     = "Fold : {}, Epoch : {}, Step : {}({:.3f}%), Loss : {:.5f}, Remaining Time : {:.3f}"\
                            .format(fold+1, epoch+1, current_step, (current_step / total_step)*100, loss.item(),
                                    (total_step - current_step)*np.mean(times))
                    logger.info(msg)
                    
                    if cfg.analysis:
                        if fold == 0:
                            wandb.log({"learning_rate":current_lr}, step=current_step)
                        wandb.log({"fold {} train loss".format(fold+1): loss.item()}, step=current_step)

                # batch end---------------------------------------------------------------        
                end             = time.perf_counter()
                times = np.append(times, end - start)
                current_step += 1
                
            #epoch end---------------------------------------------------------------------
            # validation ------------------------------------------------------------------------
            acc_table       = AccuracyTable()
            with torch.no_grad():
                valid_loss  = np.array([])
                for b in validloader:
                    _x           = b[0].to(device)
                    _target      = b[1].to(device)

                    _out         = model(_x)
                    _loss        = loss_fn(_out, _target)
                    valid_loss = np.append(valid_loss, _loss.clone().to('cpu').detach().numpy())
                    acc_table.add(_out, torch.argmax(_target, dim=1))
                    
                valid_loss_avg = np.mean(valid_loss)
                valid_loss_std = np.std(valid_loss)
                msg = "Valid\tEpoch : {}, Step : {}, valid Loss : {}".format(epoch, current_step, valid_loss_avg)
                logger.info(msg)
                
                if cfg.analysis:
                    wandb.log({"fold {} valid loss".format(fold+1): valid_loss_avg}, step=current_step)
                    wandb.log({"fold {} valid std".format(fold+1): valid_loss_std}, step=current_step)
                    acc_table.log_wandb(fold, current_step)
                    
            # save model ----------------------------------------------------- 
            if valid_loss_avg < best_loss: 
                path = Path(cfg.dir.checkpoint) / "fold{}_best.pth".format(fold+1)
                torch.save({"model":model.state_dict(),
                            "prediction":valid_loss_avg}, path)
            logger.info("epoch {} end".format(epoch))
            logger.info("---------------------------------------------------------------------------\n")
            
        if cfg.analysis:
            wandb.finish()
        
if __name__ == "__main__":
    main()