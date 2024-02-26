import numpy as np
import pywt
import polars as pl
from pathlib import Path
import os
from tqdm import tqdm

data_dir = Path("/work/abelab4/k_hiro/study/HMS/data")
paths = sorted(list(Path("/work/abelab4/k_hiro/study/HMS/data/train_eegs").glob('*.parquet')))
out_dir = Path(data_dir/"train_scalograms")
os.makedirs(out_dir, exist_ok=True)
eeg_list = ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", 
            "Pz", "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2", "EKG"]
LL_index = [0, 4, 5, 6, 7]
LP_index = [0, 1, 2, 3, 7]
RL_index = [11, 15, 16, 17, 18]
RP_index = [11, 12, 13, 14, 18]

for path  in tqdm(paths, total=len(paths)):
    eeg_id = path.stem
    eeg                      = pl.read_parquet(path)
    out_path = out_dir / "{}.parquet".format(eeg_id)
    if out_path.exists():
        continue

    eegs = np.nan_to_num(eeg[eeg_list].to_numpy())
    eegs = np.transpose(eegs)
    elem_LL = [eegs[0]-eegs[4], eegs[4]-eegs[5], eegs[5]-eegs[6], eegs[6]-eegs[7]]
    elem_LP = [eegs[0]-eegs[1], eegs[1]-eegs[2], eegs[2]-eegs[3], eegs[3]-eegs[7]]
    elem_RL = [eegs[11]-eegs[15], eegs[15]-eegs[16], eegs[16]-eegs[17], eegs[17]-eegs[18]]
    elem_RP = [eegs[11]-eegs[12], eegs[12]-eegs[13], eegs[13]-eegs[14], eegs[14]-eegs[18]]
    
    LL =  pywt.cwt(elem_LL, np.arange(1, 31), 'mexh')[0].transpose((1,0,2)).mean(axis=0)
    LP =  pywt.cwt(elem_LP, np.arange(1, 31), 'mexh')[0].transpose((1,0,2)).mean(axis=0)
    RL =  pywt.cwt(elem_RL, np.arange(1, 31), 'mexh')[0].transpose((1,0,2)).mean(axis=0)
    RP =  pywt.cwt(elem_RP, np.arange(1, 31), 'mexh')[0].transpose((1,0,2)).mean(axis=0)
    # scalograms = np.stack([LL, LP, RL, RP])
    # np.save(out_path, scalograms)

    df = pl.DataFrame([])
    for e, w in zip(['LL','LP','RL','RP'], [LL, LP, RL, RP]):
        column = ["{}_{}".format(e, i) for i in range(len(w))]
        l = pl.DataFrame(w, schema=column)
        df = df.with_columns(l)
    df.write_parquet(out_path)