from data_preparation import DataPreparation

from src.config import *

dataset_preparation = DataPreparation()
new_path = DATASET_PATH.replace(".feather", "_audio_features.feather")
dataset_preparation.calculate_audio_features(new_path=new_path, n_jobs=20, timeout=900)
