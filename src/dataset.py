import os
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset
import random
      
class BufferDataset(Dataset):
    def __init__(self, data_dir, idealized=False):
        self.data_dir = Path(data_dir)
        self.data = pl.read_csv(self.data_dir / "train.csv")
        if idealized:
            self.data = self.select_idealize(self.data)
        
    def select_idealize(self, df):
        consensus = df["expert_consensus"]
        idtfy_list = {"Seizure":9, "LPD":10, "GPD":11, "LRDA":12, "GRDA":13, "Other":14}
        
        def get_mask(x):
            _c = x[8]
            invalid_diagnosis = [v for k, v in idtfy_list.items() if k != _c]
            threshold = int(x[idtfy_list[_c]]) / 4
            if sum([x[i] for i in invalid_diagnosis]) < threshold:
                return True
            else:
                return False
        mask = df.map_rows(get_mask).get_columns()[0]
        idealized = df.filter(mask)
        return idealized
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        spc_path = self.data_dir / 'train_spectrograms'/ (str(self.data['spectrogram_id'][idx]) + '.parquet')
        eeg_path  =  self.data_dir / 'train_eegs' / (str(self.data['eeg_id'][idx]) +  '.parquet')

        spectrogram              = pl.read_parquet(spc_path)
        eeg                      = pl.read_parquet(eeg_path)
        
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


#TEST          
if __name__ == "__main__":
    
    data_dir = "./data"

    trainset = BufferDataset(data_dir, idealized=True)
    # for i in range(len(dataset)):
    #     _, c, v = dataset[i]
    #     print(i, c, v)
    #     if i == 30:
    #         break