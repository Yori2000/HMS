import glob
import os
from typing import Dict

import albumentations as A
import numpy as np
import pandas as pd
import polars as pl
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset
from utils import (AverageMeter, asMinutes, get_logger, plot_spectrogram,
                   seed_everything, sep, timeSince)

from config import Config, Paths


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



class PreProcess:
    def __init__(self, df: pl.DataFrame, config: Config, paths: Paths, label_cols: list):
        self.df = df
        self.config = config
        self.paths = paths
        self.label_cols = label_cols

    # ----------------------dataframe--------------------------------
    def get_dataframe(self):
        """Preprocess the dataframe.
        Returns: train_df (pl.DataFrame): preprocessed dataframe.
        (1) eeg_id がユニークになるように学習データを絞る (ここは拡張の余地あり)
        (2) ラベルを 0 ~ 1 に変換する
        (3) 一様分布とのkl-divergenceを計算し、それを新しい特徴量として追加する 
            (学習2段階目のデータの絞り方はもっと考えた方が良い．現状notebookどおり
            kl < 5.5を使用しているが，明らかにもっと単純なvoteの合計などで絞るべきに思える)
        # (4) 必要なカラムだけを抽出する

        """
        # (1) eeg_id がユニークになるように学習データを絞る
        train_df = self._aggregate_df(self.df)

        # (2) ラベルを 0 ~ 1 に変換する
        vote_sum = train_df.select(self.label_cols).sum_horizontal()
        train_df = train_df.with_columns(
            vote_sum.alias('vote_sum'))    # ついでにvote_sumをカラムに追加
        for column in self.label_cols:
            train_df = train_df.with_columns(
                (pl.col(column) / vote_sum).alias(column))

        # (3) 一様分布とのkl-divergenceを計算し、それを新しい特徴量として追加する
        train_df = self._add_kl_div_for_uniform_dist(train_df, self.label_cols)

        # (4) 必要なカラムだけを抽出する
        # train_df = train_df.select([
        #     'eeg_id', 'spectrogram_id', 'expert_consensus', 'vote_sum', 'kl', 'min', 'max'] + self.label_cols)

        return train_df

    def _aggregate_df(self, df: pl.DataFrame):
        """eeg_id がユニークになるように学習データを絞る.
        Args:
            df (pl.DataFrame): train.csv
        Returns:
            train_df (pl.DataFrame): preprocessed dataframe.
        """
        # eeg_id がユニークになるように学習データを絞る
        agg_df = df.group_by('eeg_id').agg(
            pl.min('spectrogram_id').alias('spectrogram_id'),
            pl.min('spectrogram_label_offset_seconds').alias('min'),
            pl.max('spectrogram_label_offset_seconds').alias('max')
        )
        train_df = df.join(agg_df, on=['eeg_id', 'spectrogram_id'])

        return train_df

    def _add_kl_div_for_uniform_dist(self, df: pl.DataFrame, label_cols: list):
        """一様分布とのkl-divergenceを計算し、それを新しい特徴量として追加する.
        Args:
            df (pl.DataFrame): preprocessed dataframe.
            label_cols (list): list of label columns.
        Returns:
            df (pl.DataFrame): preprocessed dataframe with kl-divergence.
        """
        df = add_kl(df, label_cols)

        return df

    # ----------------------spectrogram--------------------------------
    def get_spec(self):
        """Preprocess the spectrogram.
        Returns:
            kaggle_specs (Dict[int, np.ndarray]): Kaggleデータセットのスペクトログラムの辞書
            eeg_specs (Dict[int, np.ndarray]): EEGデータから作成されたスペクトログラムの辞書
        """
        # Kaggleデータセットのスペクトログラムを読み込む
        use_preloaded_kaggle_specs = self.config.USE_PRELOADED_KAGGLE_SPECS
        paths_to_kaggle_spec_files = glob.glob(
            self.paths.TRAIN_SPECTOGRAMS + '*.parquet')
        path_to_loaded_kaggle_specs = self.paths.PRE_LOADED_KAGGLE_SPECTROGRAMS

        kaggle_specs = self._read_spectrograms(
            paths_to_kaggle_spec_files,
            path_to_loaded_kaggle_specs,
            use_preloaded_kaggle_specs,
        )

        # EEGデータから作成されたスペクトログラムを読み込む
        use_preloaded_eeg_specs = self.config.USE_PRELOADED_EEG_SPECS
        paths_to_eeg_spec_files = glob.glob(
            self.paths.TRAIN_EEGS + '*.parquet')
        path_to_loaded_eeg_specs = self.paths.PRE_LOADED_EEG_SPECTROGRAMS

        eeg_specs = self._read_spectrograms(
            paths_to_eeg_spec_files,
            path_to_loaded_eeg_specs,
            use_preloaded_eeg_specs,
        )

        print(f"Kaggle specs: {len(kaggle_specs)}")
        print(f"EEG specs: {len(eeg_specs)}")
        return kaggle_specs, eeg_specs

    def _read_spectrograms(self, paths_to_specs: list, path_to_loaded_specs, use_preloaded_specs: bool):
        """Read spectrograms.
        Args:
            paths_to_specs (list): スペクトログラムのparquetファイルへのパスのリスト.
            path_to_loaded_specs (str): 保存されている(もしくはこれから保存する)npy形式のスペクトログラムへのパス
            use_preloaded_specs (bool): 事前に保存したnpy形式のスペクトログラムを使用するかどうか.
        Returns:
            specs (Dict[int, np.ndarray]): eeg_idをキーとし、スペクトログラムを値とする辞書.
        """
        if use_preloaded_specs:
            specs = np.load(path_to_loaded_specs, allow_pickle=True).item()

        else:
            specs = {}
            for path in tqdm(paths_to_specs):
                spec = pl.read_parquet(path)
                name = int(path.split('/')[-1].split('.')[0])
                specs[name] = spec[:, 1:].to_numpy()
                del spec
            np.save(path_to_loaded_specs, specs)

        return specs
    
    def get_eegs(self, paths_to_eegs: list, path_to_loaded_eegs, use_preloaded_eegs: bool):
        """Read EEGs.
        Args:
            paths_to_eegs (list): EEGデータのparquetファイルへのパスのリスト.
            path_to_loaded_eegs (str): 保存されている(もしくはこれから保存する)npy形式のEEGデータへのパス
            use_preloaded_eegs (bool): 事前に保存したnpy形式のEEGデータを使用するかどうか.
        Returns:
            eegs (Dict[int, np.ndarray]): eeg_idをキーとし、EEGデータを値とする辞書.
        """
        if use_preloaded_eegs:
            eegs = np.load(path_to_loaded_eegs, allow_pickle=True).item()

        else:
            eegs = {}
            for path in tqdm(paths_to_eegs):
                eeg = pl.read_parquet(path)
                name = int(path.split('/')[-1].split('.')[0])
                eegs[name] = eeg[:, 1:].to_numpy()
                del eeg
            np.save(path_to_loaded_eegs, eegs)

        return eegs