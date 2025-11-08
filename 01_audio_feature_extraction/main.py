from data_preparation import DataPreparation

from src.config import *

dataset_preparation = DataPreparation(DATASET_PATH)
dataset_preparation.extract_frames()
dataset_preparation.calculate_audio_features(feature_set=FEATURE_SET)
