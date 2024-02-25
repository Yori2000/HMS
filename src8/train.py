import albumentations as A
import gc
import matplotlib.pyplot as plt
import math
import multiprocessing
import numpy as np
import os
import pandas as pd
import polars as pl
import random
import time
import timm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F


from albumentations.pytorch import ToTensorV2
from glob import glob
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm
from typing import Dict, List

from utils import AverageMeter, asMinutes, timeSince, get_logger, plot_spectrogram, seed_everything, sep, add_kl
from config import Config, Paths
from dataset import CustomDataset
from models import CustomModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using', torch.cuda.device_count(), 'GPU(s)')

for path in [Paths.OUTPUT_DIR, Paths.TRAIN_SPECTOGRAMS, Paths.TRAIN_EEGS,
             Paths.PRE_LOADED_EEGS, Paths.PRE_LOADED_SPECTOGRAMS]:
    os.makedirs(os.path.dirname(path), exist_ok=True)


# 学習データの選択＋特徴量エンジニアリング
def feature_engineering(df, label_cols):
    # 
    tmp_df = df.group_by('eeg_id').agg(
        spectrogram_id=('spectrogram_id'),
        sp_label_seconds=('spectrogram_label_offset_seconds')
        )
    tmp_df = tmp_df.select(pl.col('eeg_id'), 
                          pl.col('spectrogram_id').map_elements(lambda x: min(x)), 
                          pl.col('sp_label_seconds').map_elements(lambda x: min(x)).alias('min'), 
                          pl.col('sp_label_seconds').map_elements(lambda x: max(x)).alias('max')
                          )
    
    # print(f"{tmp_df.filter(pl.col('eeg_id')==642382)=}")
    train_df = df.join(tmp_df, on='eeg_id', how='inner')
    
    train_df = train_df.unique(subset='eeg_id', keep='last')
    train_df = train_df.drop(['spectrogram_id_right', 'spectrogram_label_offset_seconds',
                              'eeg_sub_id', 'eeg_label_offset_seconds', 'spectrogram_sub_id', 
                              'spectrogram_label_offsets_seconds', 'label_id'])
    # print(f"{train_df.columns=}")
    train_df = train_df.rename({'expert_consensus':'target'})
    n = train_df.select(pl.col(r"^*_vote$")).sum_horizontal()
    for column in label_cols:
        train_df = train_df.with_columns((pl.col(column) / n).alias(column))
    # train_df = train_df.sort('eeg_id')
    train_df = add_kl(train_df, label_cols)     # testとの分布誤差調整のために追加したほうがいいらしい(?)

    return train_df

# 通常の1エポックの学習
def train_epoch(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    """One epoch training pass."""
    model.train() 
    criterion = nn.KLDivLoss(reduction="batchmean")
    scaler = torch.cuda.amp.GradScaler(enabled=Config.AMP)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    
    # ========== ITERATE OVER TRAIN BATCHES ============
    with tqdm(train_loader, unit="train_batch", desc='Train') as tqdm_train_loader:
        for step, (X, y) in enumerate(tqdm_train_loader):
            X = X.to(device)
            y = y.to(device)
            batch_size = y.size(0)
            with torch.cuda.amp.autocast(enabled=Config.AMP):
                y_preds = model(X) 
                loss = criterion(F.log_softmax(y_preds, dim=1), y)
            if Config.GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / Config.GRADIENT_ACCUMULATION_STEPS
            losses.update(loss.item(), batch_size)
            scaler.scale(loss).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            if (step + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                scheduler.step()
            end = time.time()

            # ========== LOG INFO ==========
            if step % Config.PRINT_FREQ == 0 or step == (len(train_loader)-1):
                print('Epoch: [{0}][{1}/{2}] '
                      'Elapsed {remain:s} '
                      'Loss: {loss.avg:.4f} '
                      'Grad: {grad_norm:.4f}  '
                      'LR: {lr:.8f}  '
                      .format(epoch+1, step, len(train_loader), 
                              remain=timeSince(start, float(step+1)/len(train_loader)),
                              loss=losses,
                              grad_norm=grad_norm,
                              lr=scheduler.get_last_lr()[0]))

    return losses.avg


# Mixupを使った1エポックの学習
def train_epoch_mixup(train_loader, model, criterion, optimizer, epoch, scheduler, device, mixup_alpha=0.4):
    """One epoch training pass."""
    model.train() 
    criterion = nn.KLDivLoss(reduction="batchmean")
    scaler = torch.cuda.amp.GradScaler(enabled=Config.AMP)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    
    # ========== ITERATE OVER TRAIN BATCHES ============
    with tqdm(train_loader, unit="train_batch", desc='Train') as tqdm_train_loader:
        for step, (X, y) in enumerate(tqdm_train_loader):
            X = X.to(device)
            y = y.to(device)
            
            # ========== MIXUP ==========
            lmd = np.random.beta(mixup_alpha, mixup_alpha)
            perm = torch.randperm(X.size(0)).to(device)
            X2 = X[perm, :]
            y2 = y[perm]
            X_mixup = lmd * X + (1 - lmd) * X2
            batch_size = y.size(0)
            with torch.cuda.amp.autocast(enabled=Config.AMP):
                y_preds = model(X_mixup) 
                loss = lmd * criterion(F.log_softmax(y_preds, dim=1), y) + (1 - lmd) * criterion(F.log_softmax(y_preds, dim=1), y2)
            
            if Config.GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / Config.GRADIENT_ACCUMULATION_STEPS
            losses.update(loss.item(), batch_size)
            scaler.scale(loss).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            if (step + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                scheduler.step()
            end = time.time()

            # ========== LOG INFO ==========
            if step % Config.PRINT_FREQ == 0 or step == (len(train_loader)-1):
                print('Epoch: [{0}][{1}/{2}] '
                      'Elapsed {remain:s} '
                      'Loss: {loss.avg:.4f} '
                      'Grad: {grad_norm:.4f}  '
                      'LR: {lr:.8f}  '
                      .format(epoch+1, step, len(train_loader), 
                              remain=timeSince(start, float(step+1)/len(train_loader)),
                              loss=losses,
                              grad_norm=grad_norm,
                              lr=scheduler.get_last_lr()[0]))

    return losses.avg


def valid_epoch(valid_loader, model, criterion, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    losses = AverageMeter()
    prediction_dict = {}
    preds = []
    start = end = time.time()
    with tqdm(valid_loader, unit="valid_batch", desc='Validation') as tqdm_valid_loader:
        for step, (X, y) in enumerate(tqdm_valid_loader):
            X = X.to(device)
            y = y.to(device)
            batch_size = y.size(0)
            with torch.no_grad():
                y_preds = model(X)
                loss = criterion(F.log_softmax(y_preds, dim=1), y)
            if Config.GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / Config.GRADIENT_ACCUMULATION_STEPS
            losses.update(loss.item(), batch_size)
            y_preds = softmax(y_preds)
            preds.append(y_preds.to('cpu').numpy())
            end = time.time()

            # ========== LOG INFO ==========
            if step % Config.PRINT_FREQ == 0 or step == (len(valid_loader)-1):
                print('EVAL: [{0}/{1}] '
                      'Elapsed {remain:s} '
                      'Loss: {loss.avg:.4f} '
                      .format(step, len(valid_loader),
                              remain=timeSince(start, float(step+1)/len(valid_loader)),
                              loss=losses))
                
    prediction_dict["predictions"] = np.concatenate(preds)
    return losses.avg, prediction_dict


# 2段階の学習を採用．2段階目のvalidデータは1段階目のvalidデータと同じものを使うことに注意．
# (2段階目でfilterdされたvalidデータを使うと，1段階目のvalidデータと2段階目のvalidデータの分布が異なり不当に良い結果が出るため．)
# 2段階目の学習データは1段階目の学習データのうちkl < 5.5のものを使うが，これは変更の余地あり．
# MIXUPに関する設定はconfig.pyで行う．
def train_loop(df, fold, LOGGER, target_preds, specs, eegs, debag=False):
    
    LOGGER.info(f"========== Fold: {fold} training ==========")

    # ======== SPLIT ==========
    train_folds = df.filter(pl.col('fold') != fold)
    valid_folds = df.filter(pl.col('fold') == fold)
    
    train_folds2 = train_folds.filter(pl.col('kl') < 5.5)
    valid_folds2 = valid_folds.filter(pl.col('kl') < 5.5)       # このデータは使わない．注意のために残しておく．
    
    
    # ======== DATASETS ==========
    train_dataset = CustomDataset(train_folds, Config, mode="train", augment=True, specs=specs, eeg_specs=eegs, debag=debag)
    valid_dataset = CustomDataset(valid_folds, Config, mode="train", augment=False, specs=specs, eeg_specs=eegs)
    
    train_dataset2 = CustomDataset(train_folds2, Config, mode="train", augment=True, specs=specs, eeg_specs=eegs, debag=debag)
    # valid_dataset2 = CustomDataset(valid_folds2, Config, mode="train", augment=False, specs=specs, eeg_specs=eegs)
    
    # ======== DATALOADERS ==========
    train_loader = DataLoader(train_dataset,
                              batch_size=Config.BATCH_SIZE_TRAIN,
                              shuffle=False,
                              num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=Config.BATCH_SIZE_VALID,
                              shuffle=False,
                              num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=False)
    
    train_loader2 = DataLoader(train_dataset2,
                                batch_size=Config.BATCH_SIZE_TRAIN,
                                shuffle=False,
                                num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=True)
    valid_loader2 = DataLoader(valid_dataset,
                                batch_size=Config.BATCH_SIZE_VALID,
                                shuffle=False,
                                num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=False)
    
    # =============== SET MODEL & OPTIMIZER & SCHEDULER ==========
    model = CustomModel(Config)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=Config.WEIGHT_DECAY)
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_loader)+len(train_loader2),
        pct_start=0.1,
        anneal_strategy="cos",
        final_div_factor=100,
    )

    # ======= LOSS ==========
    criterion = nn.KLDivLoss(reduction="batchmean")
    
    best_loss = np.inf
    # ====== ITERATE EPOCHS ========
    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # ======= TRAIN ==========
        if epoch < Config.MIXUP_EPOCHS:
            avg_train_loss = train_epoch_mixup(train_loader, model, criterion, optimizer, epoch, scheduler, device, mixup_alpha=Config.MIXUP_ALPHA)
        else:
            avg_train_loss = train_epoch(train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # ======= EVALUATION ==========
        avg_val_loss, prediction_dict = valid_epoch(valid_loader, model, criterion, device)
        predictions = prediction_dict["predictions"]
        
        # ======= SCORING ==========
        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_train_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            LOGGER.info(f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'predictions': predictions},
                       Paths.OUTPUT_DIR + f"/{Config.MODEL.replace('/', '_')}_fold_{fold}_best.pth")
    print(f"========== First stage training finished ==========")
    
    predictions = torch.load(Paths.OUTPUT_DIR + f"/{Config.MODEL.replace('/', '_')}_fold_{fold}_best.pth", 
                             map_location=torch.device('cpu'))['predictions']
    valid_folds[target_preds] = predictions
    
    print(f"========== Second stage training started==========")
    best_loss = np.inf
    for epoch2 in range(3):
        start_time = time.time()

        # ======= TRAIN ==========
        avg_train_loss = train_epoch(train_loader2, model, criterion, optimizer, epoch, scheduler, device)

        # ======= EVALUATION ==========
        avg_val_loss, prediction_dict = valid_epoch(valid_loader2, model, criterion, device)
        predictions = prediction_dict["predictions"]
        
        # ======= SCORING ==========
        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+epoch2+1} - avg_train_loss: {avg_train_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            LOGGER.info(f'Epoch {epoch+epoch2+1} - Save Best Loss: {best_loss:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'predictions': predictions},
                       Paths.OUTPUT_DIR + f"/{Config.MODEL.replace('/', '_')}_fold_{fold}_best.pth")
        torch.save({'model': model.state_dict(),
                    'predictions': predictions},
                    Paths.OUTPUT_DIR + f"/{Config.MODEL.replace('/', '_')}_fold_{fold}_2nd_stage_epoch_{epoch+epoch2+1}.pth")
    print(f"========== Second stage training finished ==========")

    torch.cuda.empty_cache()
    gc.collect()
    
    return valid_folds

# one-foldでの学習用．未実装．
def train_loop_full_data(df, LOGGER, target_preds):
    train_dataset = CustomDataset(df, Config, mode="train", augment=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=Config.BATCH_SIZE_TRAIN,
                              shuffle=False,
                              num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=True)
    
    model = CustomModel(Config)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=Config.WEIGHT_DECAY)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
        final_div_factor=100,
    )
    criterion = nn.KLDivLoss(reduction="batchmean")
    best_loss = np.inf
    for epoch in range(Config.EPOCHS):
        start_time = time.time()
        avg_train_loss = train_epoch(train_loader, model, criterion, optimizer, epoch, scheduler, device)
        elapsed = time.time() - start_time
        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_train_loss:.4f}  time: {elapsed:.0f}s')
        torch.save(
            {'model': model.state_dict()},
            Paths.OUTPUT_DIR + f"/{Config.MODEL.replace('/', '_')}_epoch_{epoch}.pth")
    print(f"========== Full data training finished ==========")
    torch.cuda.empty_cache()
    gc.collect()
    return _


def get_result(oof_df, label_cols, target_preds):
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    labels = torch.tensor(oof_df[label_cols].to_numpy())
    preds = torch.tensor(oof_df[target_preds].to_numpy())
    preds = F.log_softmax(preds, dim=1)
    result = kl_loss(preds, labels)
    return result


# 5-fold cross validationを行い，5つのモデルの予測を平均して提出する．
# 各モデルは異なるデータで学習するため，アンサンブルすることで精度が向上する．
# また5個のモデルの精度をもとにCVを計算するため，より正確な評価が可能となる．
def main():
    df = pl.read_csv(Paths.TRAIN_CSV)
    label_cols = df.columns[-6:]
    
    train_df = feature_engineering(df, label_cols)
    
    paths_spec = glob(Paths.TRAIN_SPECTOGRAMS+'*.parquet')
    # print(f'{len(paths_spec)=}')
    
    # (time_steps, 401)
    if Config.READ_SPEC_FILES:
        specs = {}
        for path in tqdm(paths_spec):
            aux = pl.read_parquet(path)
            name = int(path.split('/')[-1].split('.')[0])
            specs[name] = aux[:, 1:].to_numpy()
            del aux
        np.save(Paths.PRE_LOADED_SPECTOGRAMS, specs)
            
    else:
        specs = np.load(Paths.PRE_LOADED_SPECTOGRAMS, allow_pickle=True).item()
    # print(f"{specs[14960202].shape=}")
    if Config.VISUALIZE:
        idx = np.random.choice(len(paths_spec))
        plot_spectrogram(paths_spec[idx])
    
    paths_eeg = glob(Paths.TRAIN_EEGS + '*.npy')
    # print(f'{len(paths_eeg)=}')
    
    if Config.READ_EEG_SPEC_FILES:
        eegs = {}
        for path in tqdm(paths_eeg):
            eeg_spec = np.load(path)
            name = int(path.split('/')[-1].split('.')[0])
            eegs[name] = eeg_spec
            del eeg_spec

        np.save(Paths.PRE_LOADED_EEGS, eegs)
    
    else:
        eegs = np.load(Paths.PRE_LOADED_EEGS, allow_pickle=True).item()
    
    # print(f"{eegs[10249311].shape=}")
    
    # GroupKFoldを使うとシャッフルと乱数シードの設定ができないので自力実装．Kaggle本参照．
    user_id = train_df['eeg_id']
    unique_user_id = user_id.unique().to_numpy()
    fold_column = pl.Series("fold", [0]*len(train_df))
    kf = KFold(n_splits=Config.FOLDS, shuffle=True, random_state=Config.SEED)
    for fold, (train_group_idx, valid_group_idx) in enumerate(kf.split(unique_user_id)):
        train_groups = unique_user_id[train_group_idx]
        valid_groups = unique_user_id[valid_group_idx]
        
        is_train = user_id.is_in(train_groups)
        is_valid = user_id.is_in(valid_groups)
        
        # fold_column = is_train.then(fold)
        fold_column += is_valid.map_elements(lambda x: fold if x else 0)
    fold_column = fold_column.alias('fold')
    # polarsDataFrameにfold列を追加．fold列をもとに5-fold cross validationを行う．
    train_df = train_df.with_columns(fold_column)
    # print(f"{train_df.group_by('fold').agg(fold_count=pl.count('eeg_id'))=}")
    # print(f"{train_df.sort('eeg_id')=}")
    
    LOGGER = get_logger()
    target_preds = [x + "_pred" for x in ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']]
    seed_everything(Config.SEED)
    
    if not Config.TRAIN_FULL_DATA:
        oof_df = pl.DataFrame()
        for fold in range(Config.FOLDS):
            if fold in [0, 1, 2, 3, 4]:
                _oof_df = train_loop(train_df, fold, LOGGER, target_preds, specs, eegs, debag=Config.DEBUG)
                oof_df = pl.concat([oof_df, _oof_df])
                LOGGER.info(f"========== Fold {fold} result: {get_result(_oof_df, label_cols, target_preds)} ==========")
                print(f"========== Fold {fold} result: {get_result(_oof_df, label_cols, target_preds)} ==========")
        # oof_df = oof_df.reset_index(drop=True)
        LOGGER.info(f"========== CV: {get_result(oof_df, label_cols, target_preds)} ==========")
        oof_df.write_csv(Paths.OUTPUT_DIR + '/oof_df.csv')
    else:
        train_loop_full_data(train_df)


if __name__ == "__main__":
    main()