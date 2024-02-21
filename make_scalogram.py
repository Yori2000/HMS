import numpy as np
import pywt
import polars as pl
from pathlib import Path
import os

data_dir = Path("/work/abelab4/k_hiro/study/HMS/data")
csv = pl.read_csv(data_dir/"train.csv")
out_dir = Path(data_dir/"train_scalograms")
os.makedirs(out_dir, exist_ok=True)

for i  in range(len(csv)):
    data = csv[i]
    eeg_id = data['eeg_id'].item()
    eeg_path  =  data_dir / 'train_eegs' / (str(eeg_id) +  '.parquet')
    eeg                      = pl.read_parquet(eeg_path)
    eeg_list = ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", 
                "Pz", "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2", "EKG"]

    eegs = np.nan_to_num(eeg[eeg_list].to_numpy())
    eegs = np.transpose(eegs)

    wavelets, freq =  pywt.cwt(eegs, np.arange(1, 31), 'mexh')
    wavelets = np.nan_to_num(wavelets)
    wavelets = np.transpose(wavelets,(1,0,2))   # (#eeg, #frame, #bin)

    df = pl.DataFrame([])
    for e, w in zip(eeg_list, wavelets):
        column = ["{}_{}".format(e, i) for i in range(len(w))]
        l = pl.DataFrame(w, schema=column)
        df = df.with_columns(l)
    df.write_parquet(out_dir / "{}.parquet".format(eeg_id))
    