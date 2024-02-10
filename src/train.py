import os
import time

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm

from dataset import HmsTrainDataset
from models import EEGNet, SimpleModel, EEGNet2

DATETIME = time.strftime("%Y%m%d-%H%M%S")

def train_and_eval(model, train_loader, val_loader, device, fold):
    model_name = model.get_model_name()
    output_dir = f"../output/{model_name}/{DATETIME}"
    os.makedirs(output_dir, exist_ok=True)
    
    model = model.to(device)
    writer = SummaryWriter(log_dir=output_dir)
    loss_fn = nn.KLDivLoss(reduction="batchmean")
    num_epochs = 100
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    # optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    # log_interval = 1000
    checkpoint_interval = 1000
    
    step = 0
    print(f"fold {fold} Start training")
    
    best_eval_loss = float("inf")
    
    for epoch in range(num_epochs):
        for batch in tqdm(train_loader):
            x, _, y = batch
            x = x.to(device)
            # print(f"x={x}")
            y = y.to(device)
            # print(f"y={y}")
            y_hat = model(x)
            loss = KLDivLoss(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            step += 1
            # if step % log_interval == 0:
        writer.add_scalar("train_loss", loss.item(), step)
        print(f"Epoch [{epoch}/{num_epochs}], Step [{step}], train_Loss: {loss.item()}")
        eval_loss = eval(model, val_loader, device)
        writer.add_scalar("eval_loss", eval_loss, step)
        print(f"Epoch [{epoch}/{num_epochs}], Step [{step}], eval_Loss: {eval_loss.item()}")
        eval_loss = eval(model, val_loader, device)
        if eval_loss < best_eval_loss:
            torch.save(model.state_dict(), output_dir + "/" + f"checkpoint_fold{fold}.pt")
            print(f"Save {step} step model at {output_dir}")
            best_eval_loss = eval_loss

def KLDivLoss(pred, target):
    log_pred = F.log_softmax(pred, dim=1)
    return F.kl_div(log_pred, target, reduction="batchmean")

def eval(model, data_loader, device):
    model.eval()
    l_loss = []
    with torch.no_grad():
        for x, _, y in data_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = KLDivLoss(y_pred, y)
            l_loss.append(loss.item())
    return np.mean(l_loss)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    data_dir = "../data"
    csv_dir = "../data/train.csv"
    
    df = pl.read_csv(csv_dir)
    # df1 = df.unique(subset='eeg_id', keep='last')
    # df2 = df.unique(subset='eeg_id', keep='first')
    # df = pl.concat([df1,df2])
    # df = df.unique(subset=['eeg_id','eeg_sub_id'], keep='first')
    
    # kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # kf = GroupKFold(n_splits=5)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(df, df['expert_consensus'])):
        train_df = df[train_idx]
        val_df = df[val_idx]
        
        train_dataset = HmsTrainDataset(data_dir, train_df)
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
        
        val_dataset = HmsTrainDataset(data_dir, val_df)
        val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, pin_memory=True)
    
        model = EEGNet()
        # model = SimpleModel()
        train_and_eval(model, train_dataloader, val_dataloader, device, fold)
        
        break
    
    
if __name__ == "__main__":
    main()