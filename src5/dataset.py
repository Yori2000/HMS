import numpy as np
import pandas as pd
import polars as pl
import os
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict
import albumentations as A
from config import Config, Paths
from utils import plot_spectrogram
from utils import get_logger
from utils import AverageMeter
from utils import asMinutes
from utils import timeSince
from utils import seed_everything
from utils import sep


class CustomDataset(Dataset):
    def __init__(
        self, df: pl.DataFrame, 
        config,
        augment, 
        mode,
        specs,
        eeg_specs,
        debag=False
    ): 
        """Kaggleスペクトログラム(4チャンネル)とEEGデータから作成されたスペクトログラム(4チャンネル)を
        学習データとして使用するデータセット(128, 256, 8チャンネル)を作成する.

        Args:
            df (pl.DataFrame): polars dataframe.
            config (Config object): Config object.
            augment (bool): データ拡張を行うかどうか. Defaults to False.
            mode (str): 'train' or 'test'. Defaults to 'train'.
            specs (Dict[int, np.ndarray]): Kaggleデータセットのスペクトログラム. Defaults to all_spectrograms.
            eeg_specs (Dict[int, np.ndarray]): EEGデータから作成されたスペクトログラム. Defaults to all_eegs.
        """
        self.df = df
        self.config = config
        self.batch_size = self.config.BATCH_SIZE_TRAIN
        self.augment = augment
        self.mode = mode
        self.spectrograms = specs
        self.eeg_spectrograms = eeg_specs
        self.label_cols = ['seizure_vote', 'lpd_vote', 'gpd_vote', 
                           'lrda_vote', 'grda_vote', 'other_vote']
        self.debag = debag
        
        if self.debag:
            self.df = self.df.head(100)
        
    def __len__(self):
        """
        Denotes the number of batches per epoch.
        """
        return len(self.df)
        
    def __getitem__(self, index):
        """
        Generate one batch of data.
        """
        X, y = self.__data_generation(index)
        if self.augment:
            X = self.__transform(X)

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
                        
    def __data_generation(self, index):
        """
        Generates data containing batch_size samples.
        """
        X = np.zeros((128, 256, 8), dtype='float32')
        y = np.zeros(6, dtype='float32')
        img = np.ones((128,256), dtype='float32')
        row = self.df[index]
        # print(f"{row['eeg_id']=}, {row['spectrogram_id']=}")
        # print(f"{row['max'][0]=}, {row['min'][0]=}")
        if self.mode=='test': 
            r = 0
        else: 
            r = int((row['min'][0] + row['max'][0]) // 4)
            
        for region in range(4):
            img = self.spectrograms[row['spectrogram_id'][0]][r:r+300, region*100:(region+1)*100].T
            
            # Log transform spectogram
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            # Standarize per image
            ep = 1e-6
            mu = np.nanmean(img.flatten())
            std = np.nanstd(img.flatten())
            img = (img-mu)/(std+ep)
            img = np.nan_to_num(img, nan=0.0)
            X[14:-14, :, region] = img[:, 22:-22] / 2.0
        img = self.eeg_spectrograms[row['eeg_id'][0]]
        X[:, :, 4:] = img
            
        if self.mode != 'test':
            # print(f"{row[self.label_cols]=}")
            y = row[self.label_cols].to_numpy()
            y = np.ravel(y)                         # Flatten the array
        
        return X, y
    
    def __transform(self, img):
        transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
        ])
        return transforms(image=img)['image']