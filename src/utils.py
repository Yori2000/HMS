import torch
from sklearn.model_selection import KFold

import os
from pathlib import Path
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from logging import getLogger ,StreamHandler, FileHandler, Formatter
import logging
from scipy.signal import butter, lfilter
import polars as pl

class Config(dict): 
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs) 
        self.__dict__ = self 
        
        
def setup(cfg):
    
    _device = torch.device(cfg.device)
    
    log_level = logging.INFO
    if cfg.debug:
        cfg.analysis = False
        log_level = logging.DEBUG

    # working directory------------------------------------------------------------------------------
    cwd                 = Path(get_original_cwd())
    OmegaConf.set_struct(cfg, False)
    cfg.dir.checkpoint      = str(cwd / "exp" / cfg.exp_name / "checkpoint")
    cfg.dir.config          = str(cwd / "exp" / cfg.exp_name / "config")
    cfg.dir.logging         = str(cwd / "exp" / cfg.exp_name / "log")
    OmegaConf.set_struct(cfg, True)
    os.makedirs(cfg.dir.checkpoint  , exist_ok=True)
    os.makedirs(cfg.dir.config      , exist_ok=True)
    os.makedirs(cfg.dir.logging     , exist_ok=True)
    OmegaConf.save(cfg, Path(cfg.dir.config)/"params.yaml")
    # set logger-------------------------------------------------------------------------------------
    _logger = getLogger("main")
    _logger.setLevel(log_level)
    format = "%(asctime)s [%(filename)s:%(lineno)d] %(message)s"
    fl_handler = FileHandler(filename=(Path(cfg.dir.logging)/"train.log"), mode='w',encoding="utf-8")
    fl_handler.setFormatter(Formatter(format))
    fl_handler.setLevel(log_level)
    _logger.addHandler(fl_handler)
    
    return cfg, _device, _logger

class Get_cross_validation:
    """Cross validation.
    Args:
        cv_method (str): 'group_kfold' or 'kfold'.
        n_splits (int): number of splits.
        group (str): group column.
    Returns:
        folds (pl.DataFrame): folds.
    """
    def __init__(self, df: pl.DataFrame, cv_method: str, n_splits: int, group: str = None, seed: int = 0):
        self.df = df
        self.cv_method = cv_method
        self.n_splits = n_splits
        self.group = group
        self.seed = seed

    def __call__(self):
        
        if self.cv_method == 'group_kfold':
            folds = self._group_kfold(self.df, self.n_splits, self.group)
        elif self.cv_method == 'kfold':
            folds = self._kfold(self.df, self.n_splits)
        else:
            raise ValueError(f"Invalid method: {self.cv_method}")

        return folds

    def _group_kfold(self, df: pl.DataFrame, n_splits: int, group: str):
        """Group kfold.
            scikit-learnのGroupKFoldではシャッフルができないため、自前で実装する.
        Args:
            df (pl.DataFrame): dataframe.
            n_splits (int): number of splits.
            group (str): group column.
        Returns:
            folds (pl.DataFrame): folds.
        """
        unique_groups = df[group].unique().to_numpy()
        folds = pl.Series('fold', [0] * len(df))
        kf = KFold(n_splits=n_splits, shuffle=True,
                    random_state=self.seed)

        for fold, (train_index, valid_index) in enumerate(kf.split(unique_groups)):
            train_groups = unique_groups[train_index]
            valid_groups = unique_groups[valid_index]

            train_mask = df[group].is_in(train_groups)
            valid_mask = df[group].is_in(valid_groups)

            folds[valid_mask] = fold

        return folds

    def _kfold(self, df: pl.DataFrame, n_splits: int):
        """Kfold.
        Args:
            df (pl.DataFrame): dataframe.
            n_splits (int): number of splits.
        Returns:
            folds (pl.DataFrame): folds.
        """
        folds = pl.Series('fold', [0] * len(df))
        kf = KFold(n_splits=n_splits, shuffle=True,
                    random_state=self.seed)

        for fold, (train_index, valid_index) in enumerate(kf.split(df)):
            folds[valid_index] = fold

        return folds
    
def create_df(df,  n_splits):
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
             
        folds = Get_cross_validation(df=tmp, cv_method="group_kfold",
                                     n_splits=n_splits, group="eeg_id")()        
        train = tmp.with_columns(folds)
        return train

def butter_lowpass_filter(data, cutoff_freq: int = 20, sampling_rate: int = 200, order: int = 4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data

