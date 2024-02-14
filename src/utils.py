import torch

import os
from pathlib import Path
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from logging import getLogger ,StreamHandler, FileHandler, Formatter
import logging

class Config(dict): 
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs) 
        self.__dict__ = self 
        
        
def setup(cfg):
    
    OmegaConf.set_struct(cfg, False)

    _device = torch.device(cfg.device)
    
    log_level = logging.INFO
    if cfg.debug:
        torch.autograd.set_detect_anomaly(True)
        log_level = logging.DEBUG

    # working directory------------------------------------------------------------------------------
    cwd                 = Path(get_original_cwd())
    cfg.dir.checkpoint      = str(cwd / "output" / cfg.name / "checkpoint")
    cfg.dir.config          = str(cwd / "output" / cfg.name / "config")
    cfg.dir.tensorboard     = str(cwd / "output" / cfg.name / "tensorboard")
    cfg.dir.logging         = str(cwd / "output" / cfg.name / "log")
    os.makedirs(cfg.dir.checkpoint  , exist_ok=True)
    os.makedirs(cfg.dir.config      , exist_ok=True)
    os.makedirs(cfg.dir.tensorboard , exist_ok=True)
    os.makedirs(cfg.dir.logging     , exist_ok=True)
    
    # set logger-------------------------------------------------------------------------------------
    _logger = getLogger("main")
    _logger.setLevel(log_level)
    format = "%(asctime)s [%(filename)s:%(lineno)d] %(message)s"
    fl_handler = FileHandler(filename=(Path(cfg.dir.logging)/"train.log"), mode='w',encoding="utf-8")
    fl_handler.setFormatter(Formatter(format))
    fl_handler.setLevel(log_level)
    _logger.addHandler(fl_handler)
    
    OmegaConf.set_struct(cfg, True)
    
    return cfg, _device, _logger