import json
import os
import shutil
import sys
from typing import Optional, Union

import numpy as np
import pandas as pd
import scipy
import trimesh
from joblib import Parallel, delayed
from scipy.spatial.distance import euclidean
from tqdm import tqdm

sys.path.append("..")

from src.config import *
from src.models import AM, DataSetRecord
from src.paths import PathBuilder


class AMCalculator:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.df = pd.read_feather(dataset_path)
        self.landmarks = self.df[[c for c in self.df.columns if "landmark" in c]].to_numpy().reshape(-1, 68, 3)

    def calculate_ams(self, new_path: Optional[str] = None, n_jobs: int = 5, drop_na: bool = False):
        def process_single_record(index):
            try:
                landmarks = self.landmarks[index]
                return RecordAMCalculator(index=index, landmarks=landmarks).calculate_ams()
            except Exception as e:
                print(f"Error calculating AMs for {index}:\n{e}")
                return None

        all_ams = Parallel(n_jobs=n_jobs)(
            delayed(process_single_record)(index) for index in tqdm(range(len(self.landmarks)), desc="Calculating AMs")
        )
        if drop_na:
            all_ams = [calculator for calculator in all_ams if calculator is not None]

        am_df = pd.concat(all_ams, axis=0)
        self.df = self.df.merge(am_df, left_index=True, right_index=True, how="left")

        output_path = new_path if new_path is not None else self.dataset_path
        self.df.to_feather(output_path, compression="zstd", compression_level=3)


class RecordAMCalculator:
    def __init__(self, index: Union[int, str], landmarks: np.array):
        self.landmarks = landmarks
        self.index = index

    def calculate_ams(self) -> pd.DataFrame:
        ams = {}

        for am in AMS:
            ams[am.get_column_name()] = [self._calc_am(am)]

        df = pd.DataFrame(ams)
        df = df.reset_index(drop=True)
        df.index = [self.index]

        return df

    def _calc_am(self, am: AM) -> float:
        if am.type == "distance":
            return self._get_distance(self.landmarks[am.lm_indicies])
        elif am.type == "angle":
            return self._get_angle(self.landmarks[am.lm_indicies])
        else:
            return self._get_proportion(self.landmarks[am.lm_indicies])

    def _get_distance(self, lms: np.ndarray) -> float:
        return euclidean(lms[0], lms[1])

    def _get_angle(self, lms: np.ndarray) -> float:
        v1 = lms[0] - lms[1]
        v2 = lms[2] - lms[1]

        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm == 0 or v2_norm == 0:
            return 0.0

        cos_angle = np.dot(v1 / v1_norm, v2 / v2_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def _get_proportion(self, lms: np.ndarray) -> float:
        d1 = euclidean(lms[0], lms[1])
        d2 = euclidean(lms[2], lms[3])
        return d1 / d2 if d2 != 0 else 0.0
