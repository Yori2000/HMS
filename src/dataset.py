import os
import pprint

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


class HmsTrainDataset(Dataset):
    def __init__(self, data_dir, data, debag=False):
        self.data_dir = data_dir
        self.data = data
        vote_columnname = ["seizure_vote","lpd_vote","gpd_vote","lrda_vote","grda_vote","other_vote"]
        n = self.data.select(pl.col(r"^*_vote$")).sum_horizontal()
        for column in vote_columnname:
            self.data = self.data.with_columns((pl.col(column) / n).alias(column))
            
        if debag:
            self.data = self.data.head(100)


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
        
        eeg_offset_frame         = int(eeg_label_offset_seconds) * 200 
        eeg_end_frame            = eeg_offset_frame + 50 * 200
        
        eeg_list = ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", 
                    "Pz", "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2", "EKG"]
        
        eegs = [torch.from_numpy(np.nan_to_num(eeg[e][eeg_offset_frame:eeg_end_frame].to_numpy()))
                for e in eeg_list]
        eegs = torch.stack(eegs)
                
        idtfy_idx = {"Seizure":0, "LPD":1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other":5}
        expert_consensus = torch.tensor(idtfy_idx[expert_consensus])
        vote = torch.tensor([seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote])
        
        return eegs, expert_consensus, vote
        
        # return eeg, eeg_sub_id, eeg_label_offset_seconds, spectrogram, spectrogram_sub_id, \
        #     spectrogram_label_offset_seconds, label_id, patient_id, expert_consensus, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

#TEST          
if __name__ == "__main__":
    
    data_dir = "./data"
    csv_dir = "./data/train.csv"
    dataset = HmsTrainDataset(data_dir, csv_dir)
    
    data = dataset[0]
    pprint.pprint(data[0].shape)