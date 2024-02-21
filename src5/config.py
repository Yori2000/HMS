class Config:
    AMP = True
    BATCH_SIZE_TRAIN = 64
    BATCH_SIZE_VALID = 64
    EPOCHS = 4
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
    READ_SPEC_FILES = False
    READ_EEG_SPEC_FILES = False
    DEBUG = False
    
    
class Paths:
    OUTPUT_DIR = "../output/hms-efficientnetb0-pytorch-train/"
    PRE_LOADED_EEGS = '../data/brain-eeg-spectrograms/eeg_specs.npy'
    PRE_LOADED_SPECTOGRAMS = '../data/brain-spectrograms/specs.npy'
    TRAIN_CSV = "../data/train.csv"
    TRAIN_EEGS = "../data/brain-eeg-spectrograms/EEG_Spectrograms/"
    TRAIN_SPECTOGRAMS = "../data/train_spectrograms/"
