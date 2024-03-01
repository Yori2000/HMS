class Config:
    AMP = True
    BATCH_SIZE_TRAIN = 64
    BATCH_SIZE_VALID = 64
    EPOCHS = 6
    MIXUP_ALPHA = 0.4
    MIXUP_EPOCHS = 4
    FOLDS = 5
    FREEZE = False
    GRADIENT_ACCUMULATION_STEPS = 1
    MAX_GRAD_NORM = 1e7
    MODEL = "tf_efficientnet_b0"
    NUM_FROZEN_LAYERS = 39
    NUM_WORKERS = 0 # multiprocessing.cpu_count()
    PRINT_FREQ = 20
    SEED = 20
    TRAIN_FULL_DATA = False
    VISUALIZE = False
    WEIGHT_DECAY = 0.01
    USE_PRELOADED_KAGGLE_SPECS = True
    USE_PRELOADED_EEG_SPECS = True
    DEBUG = False

class Paths:
    OUTPUT_DIR = "../output/hms-efficientnetb0-pytorch-train/exp11_efficientnet_None_20hz_winlength_128_mixup_4_2_3/"
    PRE_LOADED_EEG_SPECTROGRAMS = '../data/brain-eeg-spectrograms/eeg_specs.npy'
    PRE_LOADED_KAGGLE_SPECTROGRAMS = '../data/brain-spectrograms/specs.npy'
    TRAIN_CSV = "../data/train.csv"
    TRAIN_EEGS = "../data/brain-eeg-spectrograms/EEG_Spectrograms/"
    TRAIN_SPECTOGRAMS = "../data/train_spectrograms/"
