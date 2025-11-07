import os
import subprocess
import sys
from typing import Optional, Union

import cv2
import opensmile
import pandas as pd
from tqdm import tqdm

sys.path.append("..")
from src.config import *
from src.models import DataSetRecord
from src.paths import PathBuilder


class DataPreparation:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.df = pd.read_csv(dataset_path, dtype={"clip_id": str})

    def extract_frames(self, frame_step_size: int = 10, max_num_frames: int = 10, frame_start_buffer: int = 5):
        for _, record in tqdm(self.df.iterrows(), total=len(self.df), desc="Extracting frames"):
            self._extract_frames_for_record(
                DataSetRecord(**record.to_dict()), frame_step_size, max_num_frames, frame_start_buffer
            )

    def calculate_audio_features(
        self,
        feature_set: opensmile.FeatureSet = opensmile.FeatureSet.eGeMAPSv02,
        feature_level: opensmile.FeatureLevel = opensmile.FeatureLevel.Functionals,
        new_path: Optional[str] = None,
    ):
        smile = opensmile.Smile(feature_set=feature_set, feature_level=feature_level)
        all_audio_features = []
        for index, record in tqdm(self.df.iterrows(), total=len(self.df), desc="Extracting frames"):
            try:
                all_audio_features.append(self._get_audio_features(DataSetRecord(**record.to_dict()), smile, index))
            except Exception as e:
                print(f"Error calculating audio features for {index}:\n{e}")
                continue
        audio_feature_df = pd.concat(all_audio_features, axis=0)
        self.df = self.df.merge(audio_feature_df, left_index=True, right_index=True, how="left")

        output_path = new_path if new_path is not None else self.dataset_path
        self.df.to_csv(output_path, index=False)

    def _extract_frames_for_record(
        self, record: DataSetRecord, frame_step_size: int, max_num_frames: int, frame_start_buffer: int
    ):
        paths = PathBuilder(record)
        cap = cv2.VideoCapture(paths.video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {paths.video_path}")

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - frame_start_buffer

        stop = min(frame_step_size * max_num_frames, num_frames)

        os.makedirs(paths.frame_images_path, exist_ok=True)

        for i in range(frame_start_buffer, stop, frame_step_size):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

            if not ret:
                break

            output_path = os.path.join(paths.frame_images_path, f"{paths.dataset_id}_{i}.jpg")

            if not os.path.exists(output_path):
                cv2.imwrite(output_path, frame)

    def _get_audio_features(
        self, record: DataSetRecord, smile: opensmile.Smile, index: Union[str, int]
    ) -> pd.DataFrame:
        paths = PathBuilder(record)
        self._extract_audio_from_video(paths.video_path)
        # remove old_indexes: ["file", "start", "end"]
        df = smile.process_file(TMP_AUDIO_PATH)
        df = df.reset_index(drop=True)
        df.index = [index]
        return df

    def _extract_audio_from_video(self, video_file_path: str):
        if os.path.exists(TMP_AUDIO_PATH):
            os.remove(TMP_AUDIO_PATH)

        result = subprocess.run(
            [
                "ffmpeg",
                "-i",
                video_file_path,
                "-q:a",
                "0",
                "-map",
                "a",
                TMP_AUDIO_PATH,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
