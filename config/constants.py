import os

HOME_DIR = "D:/Kaggle_Contest_Record/HMS_Harmful Brain Activity Classification"

RAW_DATA_DIR = os.path.join(HOME_DIR, "data/raw")
PROCESSED_DATA_DIR = os.path.join(HOME_DIR, "data/processed")

FEATURES_DATA_DIR = os.path.join(HOME_DIR, "features")
SUBMISSION_DIR = os.path.join(HOME_DIR, "submissions")

OOF_DIR = os.path.join(HOME_DIR, "oof")
CHECKPOINT_DIR = os.path.join(HOME_DIR, "models")

LOG_PATH = os.path.join(HOME_DIR, "tracking/training_log.csv")
#######


