import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils import Config, butter_lowpass_filter, Get_cross_validation

import hydra
from omegaconf import DictConfig
from pathlib import Path
import numpy as np
import polars as pl
from logging import getLogger
import random
import time
random.seed(0)

logger = getLogger('main').getChild('dataset')

def get_dataset(dataset_name, buf, data_dir, target):
    if dataset_name == 'CustomEEG':
        return CustomEEG(buf, data_dir, target), collate_fn
    
    
class CustomEEG(Dataset):
    def __init__(self, buf, data_dir, target):
        self.data = buf
        self.data_dir = Path(data_dir)
        self.target = target
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        eeg_id = self.data['eeg_id'][idx]
        eeg_path  =  self.data_dir / 'train_eegs' / (str(eeg_id) +  '.parquet')
        eeg                      = pl.read_parquet(eeg_path)
        
        expert_consensus         = self.data['expert_consensus'][idx]
        seizure_vote             = self.data['seizure_vote'][idx]
        lpd_vote                 = self.data['lpd_vote'][idx]
        gpd_vote                 = self.data['gpd_vote'][idx]
        lrda_vote                = self.data['lrda_vote'][idx]
        grda_vote                = self.data['grda_vote'][idx]
        other_vote               = self.data['other_vote'][idx]

        frame_offset         = int((eeg.shape[0]-10000)//2)
        frame_end            = frame_offset + 10000

        x1 = (eeg['Fp1']-eeg['T3'])[frame_offset:frame_end].to_numpy()
        x2 = (eeg['T3']-eeg['O1'])[frame_offset:frame_end].to_numpy()
        x3 = (eeg['Fp1']-eeg['C3'])[frame_offset:frame_end].to_numpy()
        x4 = (eeg['C3']-eeg['O1'])[frame_offset:frame_end].to_numpy()
        x5 = (eeg['Fp2']-eeg['C4'])[frame_offset:frame_end].to_numpy()
        x6 = (eeg['C4']-eeg['O2'])[frame_offset:frame_end].to_numpy()
        x7 = (eeg['Fp2']-eeg['T4'])[frame_offset:frame_end].to_numpy()
        x8 = (eeg['T4']-eeg['O2'])[frame_offset:frame_end].to_numpy()
        x = np.stack([x1,x2,x3,x4,x5,x6,x7,x8])
        x = np.clip(x,-1024, 1024) 
        x = np.nan_to_num(x) / 32.0
        x = butter_lowpass_filter(x)
        eegs = torch.from_numpy(x).to(torch.float)
        
        idtfy_idx = {"Seizure":0, "LPD":1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other":5}
        expert_consensus = torch.tensor(idtfy_idx[expert_consensus])
        expert_consensus = F.one_hot(expert_consensus, num_classes=6)
        vote = torch.tensor([seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote])
        
        if self.target == "vote":
            target = vote
        elif self.target == "consensus":
            target = expert_consensus
        return eegs, target
    
class Scalogram(Dataset):
    def __init__(self, data_dir, mode='train', cfg=Config({"target":"vote", 'test_size':0.2,
                                             "idealized":True, "idealized_threshould":0.75,
                                             "num_bin":30})):
        self.data_dir = Path(data_dir)
        self.target = cfg.target
        self.test_size = cfg.test_size
        self.idealized = cfg.idealized
        self.idealized_threshould = cfg.idealized_threshould
        self.num_bin = cfg.num_bin
        
        csv = pl.read_csv(self.data_dir / "train.csv")
        train, valid = self.create_df(csv)
        if mode == 'train':
            self.data = train
        if mode == 'valid':
            self.data = valid
    
    def create_df(self, df):
        VOTE = df.columns[-6:]
        train = df.group_by('eeg_id', maintain_order=True).agg(pl.min('eeg_label_offset_seconds').alias("min"))
        max_offset = df.group_by('eeg_id', maintain_order=True).agg(pl.max('eeg_label_offset_seconds').alias("max"))
        train = train.with_columns(max_offset)

        vote_sum = df.group_by('eeg_id', maintain_order=True).sum().select(['eeg_id']+VOTE)
        num_vote = vote_sum.select(VOTE).sum_horizontal()
        vote_norm = vote_sum.select(VOTE) / num_vote
        vote_norm = vote_sum.select('eeg_id').with_columns(vote_norm)
        train = train.with_columns(vote_norm)
        
        consensus = df.group_by('eeg_id', maintain_order=True).agg(pl.first('expert_consensus'))
        train = train.with_columns(consensus)
        
        vote_max = train.select(pl.col(VOTE)).max_horizontal().alias("vote_max")
        train = train.with_columns(vote_max)
        if self.idealized:
            buf = train.filter(pl.col('vote_max') > self.idealized_threshould)
            excluded = train.filter((pl.col('vote_max') <= self.idealized_threshould))   
        else:
            buf = train
            excluded = None
                     
        train = buf[:int(len(buf)*(1-self.test_size)), :]
        valid = buf[int(len(buf)*(1-self.test_size)):, :]
        if excluded:
            valid = pl.concat([valid, excluded])
        return train, valid

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        eeg_id = self.data['eeg_id'][idx]
        scalo_path  =  self.data_dir / 'train_scalograms' / (str(eeg_id) +  '.parquet')
        scalograms                      = pl.read_parquet(scalo_path)
        
        offset_min = self.data['min'][idx]
        offset_max = self.data['max'][idx]
        
        expert_consensus         = self.data['expert_consensus'][idx]
        seizure_vote             = self.data['seizure_vote'][idx]
        lpd_vote                 = self.data['lpd_vote'][idx]
        gpd_vote                 = self.data['gpd_vote'][idx]
        lrda_vote                = self.data['lrda_vote'][idx]
        grda_vote                = self.data['grda_vote'][idx]
        other_vote               = self.data['other_vote'][idx]

        frame_start         = int(offset_min) * 200 
        frame_end            = int(offset_max) + 50 * 200

        column = ["LL", "LP", "RL", "RP"]
        scalo_l = []

        for c in column:
            _scalo = np.nan_to_num(scalograms.select(pl.col("^{}_.*$".format(c)))[frame_start:frame_end].to_numpy())
            scalo_l.append(_scalo)

        scalograms = np.stack(scalo_l)
        scalograms = torch.from_numpy(scalograms).to(torch.float)

        idtfy_idx = {"Seizure":0, "LPD":1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other":5}
        expert_consensus = torch.tensor(idtfy_idx[expert_consensus])
        expert_consensus = F.one_hot(expert_consensus, num_classes=6)
        vote = torch.tensor([seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote])
        
        if self.target == "vote":
            target = vote
        elif self.target == "consensus":
            target = expert_consensus
        return scalograms, target

def collate_fn(batch):
    x = [b[0] for b in batch]
    target = [b[1] for b in batch]
    
    max_length = max([_x.shape[1] for _x in x])
    x = [F.pad(_x, (0, max_length - _x.shape[1]),"constant",0) for _x in x]
    x = torch.stack(x)
    target = torch.stack(target)
    
    return x, target

def collate_fn_2d(batch):

    x = [b[0] for b in batch]
    target = [b[1] for b in batch]
    
    max_length = max([_x.shape[1] for _x in x])

    x = [F.pad(_x, (0, 0, 0, max_length - _x.shape[1]),"constant",0) for _x in x]
    x = torch.stack(x)
    target = torch.stack(target)
    
    return x, target


#TEST          
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg : DictConfig):
    
    trainset = CustomEEG(cfg.dir.input, 'train', cfg.dataset)
    trainloader  = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True,collate_fn=collate_fn)
    for i, b in enumerate(trainloader):
        x, t= b
        print(i, x.shape)
        if i == 3:
            break

if __name__ == "__main__":
    main()
