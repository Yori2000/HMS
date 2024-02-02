import os

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


class HmsTrainDataset(Dataset):
    def __init__(self, data_dir, filelist_csv):
        self.data_dir = data_dir
        self.data = pl.read_csv(filelist_csv)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        spectrogram              = pl.read_parquet(os.path.join(
                                            self.data_dir, 'train_spectrograms', 
                                            str(self.data['spectrogram_id'][idx]) + '.parquet'))
        
        eeg                      = pl.read_parquet(os.path.join(
                                            self.data_dir, 'train_eegs', 
                                            (str(self.data['eeg_id'][idx]) +  '.parquet')))
        
        eeg_sub_id               = self.data['eeg_sub_id'][idx]
        eeg_label_offset_seconds = self.data['eeg_label_offset_seconds'][idx]
        spectrogram_sub_id       = self.data['spectrogram_sub_id'][idx]
        spectrogram_label_offset_seconds = self.data['spectrogram_label_offset_seconds'][idx]
        label_id                 = self.data['label_id'][idx]
        patient_id               = self.data['patient_id'][idx]
        expert_consensus         = self.data['expert_consensus'][idx]
        seizure_vote             = self.data['seizure_vote'][idx]
        lpd_vote                 = self.data['lpd_vote'][idx]
        gpd_vote                 = self.data['gpd_vote'][idx]
        lrda_vote                = self.data['lrda_vote'][idx]
        grda_vote                = self.data['grda_vote'][idx]
        other_vote               = self.data['other_vote'][idx]

        return eeg, eeg_sub_id, eeg_label_offset_seconds, spectrogram, spectrogram_sub_id, \
            spectrogram_label_offset_seconds, label_id, patient_id, expert_consensus, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote
  
  
#TEST          
if __name__ == "__main__":
    
    data_dir = "./data"
    csv_dir = "./data/train.csv"
    dataset = HmsTrainDataset(data_dir, csv_dir)
    
    data = dataset[0]
    
    print(data)