import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils import Config

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

def get_dataset(data_dir, mode, cfg):
    if cfg.dataset_name =='EEG':
        return EEG(data_dir, mode, cfg), collate_fn
    elif cfg.dataset_name == 'NonOverlapEEG':
        return NonOverlapEEG(data_dir, mode, cfg), collate_fn
    elif cfg.dataset_name == 'CustomEEG':
        return CustomEEG(data_dir, mode, cfg), collate_fn
    elif cfg.dataset_name == 'Scalogram':
        return Scalogram(data_dir, mode, cfg), collate_fn_2d

class EEG(Dataset):
    def __init__(self, data_dir, mode='train', cfg=Config({"target":"vote", 'test_size':0.2,
                                   "idealized":True, "idealized_threshould":0.75,
                                   "exclude_Other":False})):
        self.data_dir = Path(data_dir)
        self.test_size = cfg.test_size
        self.idealized = cfg.idealized
        self.target   = cfg.target
        self.idealized_threshould = cfg.idealized_threshould
        self.exclude_other = cfg.exclude_other
        
        csv = pl.read_csv(self.data_dir / "train.csv")
        train, valid = self.create_df(csv)
        if mode == 'train':
            self.data = train
        if mode == 'valid':
            self.data = valid
    
    def create_df(self, df):
        VOTE = df.columns[-6:]
        tmp = df.group_by('eeg_id', maintain_order=True).agg(
            pl.first('expert_consensus'),
            pl.first('patient_id'),
            pl.sum(VOTE))
        num_vote = tmp.select(VOTE).sum_horizontal()
        vote_norm = tmp.select(VOTE) / num_vote
        tmp = tmp.with_columns(vote_norm)
        vote_max = tmp.select(pl.col(VOTE)).max_horizontal().alias("vote_max")
        tmp = tmp.with_columns(vote_max)
        if self.idealized:
            buf = tmp.filter(pl.col('vote_max') > self.idealized_threshould)
            excluded = tmp.filter((pl.col('vote_max') <= self.idealized_threshould))   
        elif self.exclude_other:
            buf = tmp.filter(pl.col('expert_consensus') != 'Other')
            excluded = tmp.filter((pl.col('expert_consensus') == 'Other'))
        elif self.idealized and self.exclude_other:
            buf = tmp.filter((pl.col('vote_max') > self.idealized_threshould) |
                                  (pl.col('expert_consensus') != 'Other'))
            excluded = tmp.filter((pl.col('vote_max') <= self.idealized_threshould) |
                                  (pl.col('expert_consensus') == 'Other'))
        else:
            buf = tmp
            excluded = None
                     
        train = buf[:int(len(buf)*(1-self.test_size)), :]
        valid = pl.concat([buf[int(len(buf)*(1-self.test_size)):, :], excluded])
        return train, valid
    
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
        
        eeg_list = ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", 
                    "Pz", "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2", "EKG"]
        
        eegs = np.nan_to_num(eeg[eeg_list][frame_offset:frame_end].to_numpy())
        eegs = np.transpose(eegs)
        eegs = torch.from_numpy(eegs)

        idtfy_idx = {"Seizure":0, "LPD":1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other":5}
        expert_consensus = torch.tensor(idtfy_idx[expert_consensus])
        expert_consensus = F.one_hot(expert_consensus, num_classes=6)
        
        vote = torch.tensor([seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote])
        
        if self.target == "vote":
            target = vote
        elif self.target == "consensus":
            target = expert_consensus
        return eegs, target

    
class NonOverlapEEG(Dataset):
    def __init__(self, data_dir, mode, cfg=Config({"target":"vote", 'test_size':0.2,
                                   "idealized":True, "idealized_threshould":0.75})):
        self.data_dir = Path(data_dir)
        self.test_size = cfg.test_size
        self.idealized = cfg.idealized
        self.target = cfg.target
        
        csv = pl.read_csv(self.data_dir / "train.csv")
        self.data = self.create_df(csv)
        train, valid = self.create_df(csv)
        if mode == 'train':
            self.data = train
        if mode == 'valid':
            self.data = valid
        self.max_offset = self.data.select(pl.col("max")).max()
        
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
        valid = pl.concat([buf[int(len(buf)*(1-self.test_size)):, :], excluded])
        return train, valid
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        eeg_id = self.data['eeg_id'][idx]
        eeg_path  =  self.data_dir / 'train_eegs' / (str(eeg_id) +  '.parquet')
        eeg                      = pl.read_parquet(eeg_path)
        
        eeg_offset_min = self.data['min'][idx]
        eeg_offset_max = self.data['max'][idx]
        
        expert_consensus         = self.data['expert_consensus'][idx]
        seizure_vote             = self.data['seizure_vote'][idx]
        lpd_vote                 = self.data['lpd_vote'][idx]
        gpd_vote                 = self.data['gpd_vote'][idx]
        lrda_vote                = self.data['lrda_vote'][idx]
        grda_vote                = self.data['grda_vote'][idx]
        other_vote               = self.data['other_vote'][idx]

        eeg_offset_frame         = int(eeg_offset_min * 200)
        eeg_end_frame            = int((eeg_offset_max + 50) * 200)
        
        eeg_list = ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", 
                    "Pz", "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2", "EKG"]
        
        eegs = np.nan_to_num(eeg[eeg_list][eeg_offset_frame:eeg_end_frame].to_numpy())
        eegs = np.transpose(eegs)
        eegs = torch.from_numpy(eegs)

        idtfy_idx = {"Seizure":0, "LPD":1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other":5}
        expert_consensus = torch.tensor(idtfy_idx[expert_consensus])
        expert_consensus = F.one_hot(expert_consensus, num_classes=6)
        vote = torch.tensor([seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote])
        
        if self.target == "vote":
            target = vote
        elif self.target == "consensus":
            target = expert_consensus
        return eegs, target

class CustomEEG(Dataset):
    def __init__(self, data_dir, mode='train', cfg=Config({"target":"consensus","test_size":0.2,
                                   "idealized":True, "idealized_threshould":0.75,
                                   "exclude_other":False})):
        self.data_dir = Path(data_dir)
        self.test_size = cfg.test_size
        self.idealized = cfg.idealized
        self.idealized_threshould = cfg.idealized_threshould
        self.exclude_other = cfg.exclude_other
        self.target = cfg.target
        
        csv = pl.read_csv(self.data_dir / "train.csv")
        train, valid = self.create_df(csv)
        if mode == 'train':
            self.data = train
        if mode == 'valid':
            self.data = valid
        
    def create_df(self, df):
        VOTE = df.columns[-6:]
        tmp = df.group_by('eeg_id', maintain_order=True).agg(
            pl.first('expert_consensus'),
            pl.first('patient_id'),
            pl.sum(VOTE))
        num_vote = tmp.select(VOTE).sum_horizontal()
        vote_norm = tmp.select(VOTE) / num_vote
        tmp = tmp.with_columns(vote_norm)
        vote_max = tmp.select(pl.col(VOTE)).max_horizontal().alias("vote_max")
        tmp = tmp.with_columns(vote_max)
        
        if self.idealized:
            buf = tmp.filter(pl.col('vote_max') > self.idealized_threshould)
            excluded = tmp.filter((pl.col('vote_max') <= self.idealized_threshould))   
        elif self.exclude_other:
            buf = tmp.filter(pl.col('expert_consensus') != 'Other')
            excluded = tmp.filter((pl.col('expert_consensus') == 'Other'))
        elif self.idealized and self.exclude_other:
            buf = tmp.filter((pl.col('vote_max') > self.idealized_threshould) |
                                  (pl.col('expert_consensus') != 'Other'))
            excluded = tmp.filter((pl.col('vote_max') <= self.idealized_threshould) |
                                  (pl.col('expert_consensus') == 'Other'))
        else:
            buf = tmp
            excluded = None
                     
        train = buf[:int(len(buf)*(1-self.test_size)), :]
        valid = pl.concat([buf[int(len(buf)*(1-self.test_size)):, :], excluded])
        return train, valid
    
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
        eeg_list = ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", 
                    "Pz", "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2", "EKG"]
        eegs = np.nan_to_num(eeg[eeg_list][frame_offset:frame_end].to_numpy())
        eegs = np.transpose(eegs)
        LL = np.stack([eegs[0]-eegs[4], eegs[4]-eegs[5], eegs[5]-eegs[6], eegs[6]-eegs[7]])
        LP = np.stack([eegs[0]-eegs[1], eegs[1]-eegs[2], eegs[2]-eegs[3], eegs[3]-eegs[7]])
        RL = np.stack([eegs[11]-eegs[15], eegs[15]-eegs[16], eegs[16]-eegs[17], eegs[17]-eegs[18]])
        RP = np.stack([eegs[11]-eegs[12], eegs[12]-eegs[13], eegs[13]-eegs[14], eegs[14]-eegs[18]] )       
        eegs = torch.from_numpy(np.stack([LL,LP,RL,RP]))
        
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
        valid = pl.concat([buf[int(len(buf)*(1-self.test_size)):, :], excluded])
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
