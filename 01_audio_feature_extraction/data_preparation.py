import os
import signal
import subprocess
import sys
import threading
from typing import Optional, Union

import opensmile
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.append("..")
from src.config import *
from src.models import DataSetRecord
from src.paths import PathBuilder


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException()


class DataPreparation:
    def __init__(self):
        self.df = pd.read_feather(DATASET_PATH)

    def calculate_audio_features(
        self, n_jobs: int = 10, drop_na: bool = True, new_path: Optional[str] = None, timeout: int = 60
    ):
        def get_audio_features(
            record_tuple,
        ):
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout)
                index, record_dict = record_tuple
                record = DataSetRecord(**record_dict)
                return AudioFeatureExtractor(record, index).get_audio_features()
            except TimeoutException:
                worker_id = threading.current_thread().name
                print(f"Timeout processing record for {index} {record.clip_id} on worker {worker_id}!")
                return None
            except Exception as e:
                print(f"Error calculating audio features for {index} {record.clip_id}:\n{e}")
                return None

        records_to_process = [(index, record.to_dict()) for index, record in self.df.iterrows()]

        all_audio_features = Parallel(n_jobs=n_jobs)(
            delayed(get_audio_features)(record_data)
            for record_data in tqdm(records_to_process, desc="Calculating Audio Features")
        )

        print("Dropping Na:")
        if drop_na:
            all_audio_features = [f for f in all_audio_features if f is not None]

        print("Concatinating Dicts")
        audio_feature_df = pd.concat(all_audio_features, axis=0)
        self.df = self.df.merge(audio_feature_df, left_index=True, right_index=True, how="left")

        print("Resetting Index")
        self.df = self.df.reset_index(drop=True)

        print("Saving to Feather...")
        output_path = new_path if new_path is not None else self.dataset_path
        self.df.to_feather(output_path, compression="zstd", compression_level=3)


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
        if not os.path.exists(self.paths.video_path):
            raise FileNotFoundError

        result = subprocess.run(
            [
                "ffmpeg",
                "-i",
                self.paths.video_path,
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
        if not os.path.exists(self.paths.video_path):
            raise FileNotFoundError
