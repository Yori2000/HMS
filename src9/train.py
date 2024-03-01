from typing import Any
import torch
import polars as pl
import numpy as np
import glob
import tqdm
from sklearn.model_selection import KFold

from config import Config, Paths
from dataset import CustomDataset, PreProcess
from utils import get_logger, AverageMeter, asMinutes, timeSince, plot_spectrogram, seed_everything, sep, add_kl


class Run():
    def __init__(self, config: Config, paths: Paths):
        self.config = config
        self.paths = paths
        self.logger = get_logger()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Device: {self.device}")



# ----------------------cross_validation--------------------------------
class Get_cross_validation:
    """Cross validation.
    Args:
        cv_method (str): 'group_kfold' or 'kfold'.
        n_splits (int): number of splits.
        group (str): group column.
    Returns:
        folds (pl.DataFrame): folds.
    """
    def __init__(self, df: pl.DataFrame, cv_method: str, n_splits: int, group: str = None):
        self.df = df
        self.cv_method = cv_method
        self.n_splits = n_splits
        self.group = group

    def __call__(self):
        print(f'{self.df.shape=}, {self.cv_method=}, {self.n_splits=}, {self.group=}')
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
        print(df)
        unique_groups = df[group].unique().to_numpy()
        folds = pl.Series('fold', [0] * len(df))
        kf = KFold(n_splits=n_splits, shuffle=True,
                    random_state=self.config.SEED)

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
                    random_state=self.config.SEED)

        for fold, (train_index, valid_index) in enumerate(kf.split(df)):
            folds[valid_index] = fold

        return folds


if __name__ == '__main__':
    df = pl.read_csv(Paths.TRAIN_CSV)
    # ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
    label_cols = df.columns[-6:]

    dataprep = PreProcess(df, Config, Paths, label_cols)
    train_df = dataprep.get_dataframe()
    kaggle_specs, eeg_specs = dataprep.get_spec()

    folds = Get_cross_validation(
        df=train_df, cv_method="group_kfold", n_splits=Config.FOLDS, group="eeg_id")()
    train_df = train_df.with_columns(folds)
    print(train_df)

    # train = Train(train_df, kaggle_specs, eeg_specs, Config, Paths)
    # train.run()
