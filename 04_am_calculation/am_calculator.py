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
        self.df = pd.read_csv(dataset_path, dtype={"clip_id": str})

    def calculate_ams(
        self, delete_interm_files: bool = False, new_path: Optional[str] = None, n_jobs: int = 5, drop_na: bool = False
    ):
        def process_single_record(record_tuple):
            index, record_dict, delete = record_tuple
            try:
                record_obj = DataSetRecord(**record_dict)
                return RecordAMCalculator(record=record_obj, index=index, delete=delete).calculate_ams()
            except Exception as e:
                print(f"Error calculating AMs for {index}:\n{e}")
                return None

        records_to_process = [(index, record.to_dict(), delete_interm_files) for index, record in self.df.iterrows()]

        all_ams = Parallel(n_jobs=n_jobs)(
            delayed(process_single_record)(record_data)
            for record_data in tqdm(records_to_process, desc="Calculating AMs")
        )
        if drop_na:
            all_ams = [calculator for calculator in all_ams if calculator is not None]

        audio_feature_df = pd.concat(all_ams, axis=0)
        self.df = self.df.merge(audio_feature_df, left_index=True, right_index=True, how="left")

        output_path = new_path if new_path is not None else self.dataset_path
        self.df.to_csv(output_path, index=False)


class RecordAMCalculator:
    def __init__(self, record: DataSetRecord, index: Union[str, int], delete: bool = True):
        self.record = record
        self.index = index
        self.delete = delete

        bfm_model_front = scipy.io.loadmat(os.path.join(BFM_PATH, "BFM_model_front.mat"))
        self.landmark_indicies: list = bfm_model_front["keypoints"].flatten() - 1

        self.paths = PathBuilder(self.record)
        self.landmarks = [
            self._get_landmarks(self.paths.result_0_mesh),
            self._get_landmarks(self.paths.result_1_mesh),
            self._get_landmarks(self.paths.result_2_mesh),
        ]

    def calculate_ams(self) -> pd.DataFrame:
        ams = {}

        for am in AMS:
            ams[am.get_column_name()] = [self._calc_avg_am(am)]

        df = pd.DataFrame(ams)
        df = df.reset_index(drop=True)
        df.index = [self.index]

        if self.delete:
            self._delete_interm_files()

        return df

    def _delete_interm_files(self):
        folders_to_delete = [self.paths.frame_images_path]
        files_to_delete = [
            self.paths.image_0_path,
            self.paths.image_1_path,
            self.paths.image_2_path,
            self.paths.image_detection_0_path,
            self.paths.image_detection_1_path,
            self.paths.image_detection_2_path,
            self.paths.result_0_coefficients,
            self.paths.result_0_image,
            self.paths.result_0_mesh,
            self.paths.result_1_coefficients,
            self.paths.result_1_image,
            self.paths.result_1_mesh,
            self.paths.result_2_coefficients,
            self.paths.result_2_image,
            self.paths.result_2_mesh,
        ]

        for folder in folders_to_delete:
            if os.path.exists(folder):
                shutil.rmtree(folder)

        for file in files_to_delete:
            if os.path.exists(file):
                os.remove(file)

    def _get_landmarks(self, mesh_path: str) -> np.array:
        mesh: trimesh.Geometry = trimesh.load(mesh_path)
        vertices_3d = mesh.vertices

        return np.array([np.array(v) for v in vertices_3d[self.landmark_indicies.astype(int)]])

    def _calc_avg_am(self, am: AM):
        return np.average([self._calc_am(lm, am) for lm in self.landmarks])

    def _calc_am(self, landmarks: np.array, am: AM) -> float:
        if am.type == "distance":
            return self._get_distance(landmarks[am.lm_indicies])
        elif am.type == "angle":
            return self._get_angle(landmarks[am.lm_indicies])
        else:
            return self._get_proportion(landmarks[am.lm_indicies])

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
