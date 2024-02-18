import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import os
from pathlib import Path
import numpy as np
import polars as pl
from logging import getLogger
import pywt

logger = getLogger('main').getChild('dataset')

def get_dataset(cfg):
    if cfg.dataset =='EEGDataset':
        return EEGDataset(cfg.dir.input), collate_fn
    elif cfg.dataset == 'NonOverlapDataset':
        return NonOverlapDataset(cfg.dir.input), collate_fn
    elif cfg.dataset == 'WaveletDataset':
        return WaveletDataset(cfg.dir.input), None

class EEGDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        csv = pl.read_csv(self.data_dir / "train.csv")
        self.data = self.create_df(csv)
    
    def create_df(self, df):
        VOTE = df.columns[-6:]
        train = df.select('eeg_id', 'eeg_label_offset_seconds','expert_consensus')
        
        num_vote = df.select(VOTE).sum_horizontal()
        vote_norm = df.select(VOTE) / num_vote
        train = train.with_columns(vote_norm)
        
        vote_max = train.select(pl.col(VOTE)).max_horizontal().alias("vote_max")
        train = train.with_columns(vote_max)
        train = train.filter(pl.col('vote_max') > 0.7)

        return train
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        eeg_id = self.data['eeg_id'][idx]
        eeg_path  =  self.data_dir / 'train_eegs' / (str(eeg_id) +  '.parquet')
        eeg                      = pl.read_parquet(eeg_path)
        
        eeg_label_offset_seconds = self.data['eeg_label_offset_seconds'][idx]
        expert_consensus         = self.data['expert_consensus'][idx]
        seizure_vote             = self.data['seizure_vote'][idx]
        lpd_vote                 = self.data['lpd_vote'][idx]
        gpd_vote                 = self.data['gpd_vote'][idx]
        lrda_vote                = self.data['lrda_vote'][idx]
        grda_vote                = self.data['grda_vote'][idx]
        other_vote               = self.data['other_vote'][idx]

        eeg_offset_frame         = int(eeg_label_offset_seconds) * 200 
        eeg_end_frame            = eeg_offset_frame + 50 * 200
        
        eeg_list = ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", 
                    "Pz", "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2", "EKG"]
        
        eegs = np.nan_to_num(eeg[eeg_list][eeg_offset_frame:eeg_end_frame].to_numpy())
        eegs = np.transpose(eegs)
        eegs = torch.from_numpy(eegs)

        idtfy_idx = {"Seizure":0, "LPD":1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other":5}
        expert_consensus = torch.tensor(idtfy_idx[expert_consensus])
        vote = torch.tensor([seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote])
        
        return eegs, expert_consensus, vote

    
class NonOverlapDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        csv = pl.read_csv(self.data_dir / "train.csv")
        self.data = self.create_df(csv)
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
        train = train.filter(pl.col('vote_max') > 0.75)
        
        train = train.filter(pl.col('max') < 500)
        return train
    
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
        
        eegs = [torch.from_numpy(np.nan_to_num(eeg[e][eeg_offset_frame:eeg_end_frame].to_numpy()))
                for e in eeg_list]
        eegs = torch.stack(eegs)

        idtfy_idx = {"Seizure":0, "LPD":1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other":5}
        expert_consensus = torch.tensor(idtfy_idx[expert_consensus])
        vote = torch.tensor([seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote])
        
        return eegs, expert_consensus, vote

class WaveletDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        csv = pl.read_csv(self.data_dir / "train.csv")
        self.data = self.create_df(csv)
    
    def create_df(self, df):
        VOTE = df.columns[-6:]
        train = df.select('eeg_id', 'eeg_label_offset_seconds','expert_consensus')
        
        num_vote = df.select(VOTE).sum_horizontal()
        vote_norm = df.select(VOTE) / num_vote
        train = train.with_columns(vote_norm)
        
        vote_max = train.select(pl.col(VOTE)).max_horizontal().alias("vote_max")
        train = train.with_columns(vote_max)
        train = train.filter(pl.col('vote_max') > 0.7)

        return train
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        eeg_id = self.data['eeg_id'][idx]
        eeg_path  =  self.data_dir / 'train_eegs' / (str(eeg_id) +  '.parquet')
        eeg                      = pl.read_parquet(eeg_path)
        
        eeg_label_offset_seconds = self.data['eeg_label_offset_seconds'][idx]
        expert_consensus         = self.data['expert_consensus'][idx]
        seizure_vote             = self.data['seizure_vote'][idx]
        lpd_vote                 = self.data['lpd_vote'][idx]
        gpd_vote                 = self.data['gpd_vote'][idx]
        lrda_vote                = self.data['lrda_vote'][idx]
        grda_vote                = self.data['grda_vote'][idx]
        other_vote               = self.data['other_vote'][idx]

        eeg_offset_frame         = int(eeg_label_offset_seconds) * 200 
        eeg_end_frame            = eeg_offset_frame + 50 * 200
        
        eeg_list = ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", 
                    "Pz", "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2", "EKG"]
        
        eegs = np.nan_to_num(eeg[eeg_list][eeg_offset_frame:eeg_end_frame].to_numpy())
        eegs = np.transpose(eegs)
        
        wavelets, freq =  pywt.cwt(eegs, np.arange(1, 31), 'mexh')
        wavelets = np.transpose(wavelets,(1,2,0))   # (#eegs, #frame, #bin)
        wavelets = torch.from_numpy(wavelets)
        
        idtfy_idx = {"Seizure":0, "LPD":1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other":5}
        expert_consensus = torch.tensor(idtfy_idx[expert_consensus])
        vote = torch.tensor([seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote])
        
        return wavelets, expert_consensus, vote

def collate_fn(batch):
    eegs = [b[0] for b in batch]
    expert_consensus = [b[1] for b in batch]
    vote = [b[2] for b in batch]
    
    max_length = max([eeg.shape[1] for eeg in eegs])
    eegs = [F.pad(eeg, (0, max_length - eeg.shape[1]),"constant",0) for eeg in eegs]
    eegs = torch.stack(eegs)
    expert_consensus= torch.stack(expert_consensus)
    vote = torch.stack(vote)
    #debug shape
    logger.debug("eeg shape : {}".format(eegs.shape))
    
    return eegs, expert_consensus, vote
#TEST          
if __name__ == "__main__":
    
    data_dir = "./data"

    trainset = WaveletDataset(data_dir)
    trainloader  = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True,collate_fn=None)

    for i, b in enumerate(trainloader):
        e, c, v = b
        print(i, e.shape, c, v)
        if i == 1:
            break