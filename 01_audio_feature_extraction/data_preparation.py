import os
import subprocess
import sys
from typing import Optional, Union

import opensmile
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.append("..")
from src.config import *
from src.models import DataSetRecord
from src.paths import PathBuilder


class DataPreparation:
    def __init__(self):
        self.df = pd.read_csv(DATASET_PATH, dtype={"clip_id": str})

    def calculate_audio_features(
        self,
        n_jobs: int = 10,
        drop_na: bool = True,
        new_path: Optional[str] = None,
    ):
        def get_audio_features(record_tuple):
            try:
                record_dict, index = record_tuple
                record = DataSetRecord(**record_dict)
                return AudioFeatureExtractor(record, index).get_audio_features()
            except Exception as e:
                return None

        records_to_process = [(index, record.to_dict()) for index, record in self.df.iterrows()]

        all_audio_features = Parallel(n_jobs=n_jobs)(
            delayed(get_audio_features)(record_data)
            for record_data in tqdm(records_to_process, desc="Calculating AMs")
        )
        if drop_na:
            all_audio_features = [calculator for calculator in all_audio_features if calculator is not None]

        audio_feature_df = pd.concat(all_audio_features, axis=0)
        self.df = self.df.merge(audio_feature_df, left_index=True, right_index=True, how="left")

        output_path = new_path if new_path is not None else self.dataset_path
        self.df.to_csv(output_path, index=False)


class AudioFeatureExtractor:
    def __init__(
        self,
        record: DataSetRecord,
        index: Union[str, int],
    ):
        self.record = record
        self.smile = opensmile.Smile(feature_set=FEATURE_SET, feature_level=FEATURE_LEVEL)
        self.paths = PathBuilder(record)
        self.index = index

    def get_audio_features(
        self,
    ) -> pd.DataFrame:
        self._extract_audio_from_video()
        # remove old_indexes: ["file", "start", "end"]
        df = self.smile.process_file(self.paths.tmp_audio_path)
        df = df.reset_index(drop=True)
        df.index = [self.index]
        os.remove(self.paths.tmp_audio_path)
        return df

    def _extract_audio_from_video(self):
        result = subprocess.run(
            [
                "ffmpeg",
                "-i",
                self.paths.video_file_path,
                "-q:a",
                "0",
                "-map",
                "a",
                self.paths.tmp_audio_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
