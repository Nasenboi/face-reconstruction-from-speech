import json
import os
import shutil
import sys
from typing import Optional, Union

import numpy as np
import pandas as pd
import scipy
import trimesh
from scipy.spatial.distance import euclidean
from tqdm import tqdm

sys.path.append("..")

from src.config import *
from src.models import AM, DataSetRecord
from src.paths import PathBuilder

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lm_map_path = os.path.join(parent_dir, "am_map.json")
AM_MAP: dict = json.load(open(lm_map_path))


class AMCalculator:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.df = pd.read_csv(dataset_path, dtype={"clip_id": str})

        bfm_model_front = scipy.io.loadmat(os.path.join(BFM_PATH, "BFM_model_front.mat"))
        self.landmark_indicies: list = bfm_model_front["keypoints"].flatten() - 1

    def calculate_ams(self, delete_interm_files: bool = False, new_path: Optional[str] = None):
        all_ams = []
        for index, record in tqdm(self.df.iterrows(), total=len(self.df), desc="Extracting frames"):
            try:
                all_ams.append(
                    self._calculate_record_ams(DataSetRecord(**record.to_dict()), index, delete_interm_files)
                )
            except Exception as e:
                print(f"Error calculating ams for {index}:\n{e}")
                continue
        audio_feature_df = pd.concat(all_ams, axis=0)
        self.df = self.df.merge(audio_feature_df, left_index=True, right_index=True, how="left")

        output_path = new_path if new_path is not None else self.dataset_path
        self.df.to_csv(output_path, index=False)

    def _calculate_record_ams(self, record: DataSetRecord, index: Union[str, int], delete: bool) -> pd.DataFrame:
        paths = PathBuilder(record)
        landmarks_0 = self._get_landmarks(paths.result_0_mesh)
        landmarks_1 = self._get_landmarks(paths.result_1_mesh)
        landmarks_2 = self._get_landmarks(paths.result_2_mesh)
        ams = {}

        for dist_am in AM_MAP["distance"]:
            am = AM(type="distance", lm_indicies=dist_am)
            ams[f"distance_{dist_am[0]:02d}_{dist_am[1]:02d}"] = [
                self._calc_avg_am(landmarks_0, landmarks_1, landmarks_2, am)
            ]
        for angle_am in AM_MAP["angle"]:
            am = AM(type="angle", lm_indicies=angle_am)
            ams[f"angle_{angle_am[0]:02d}_{angle_am[1]:02d}_{angle_am[2]:02d}"] = [
                self._calc_avg_am(landmarks_0, landmarks_1, landmarks_2, am)
            ]
        for prop_am in AM_MAP["proportion"]:
            am = AM(type="proportion", lm_indicies=prop_am)
            ams[f"proportion_{prop_am[0]:02d}_{prop_am[1]:02d}_{prop_am[2]:02d}_{prop_am[1]:02d}"] = [
                self._calc_avg_am(landmarks_0, landmarks_1, landmarks_2, am)
            ]

        df = pd.DataFrame(ams)
        df = df.reset_index(drop=True)
        df.index = [index]

        if delete:
            self._delete_interm_files(paths)

        return df

    def _delete_interm_files(self, paths: PathBuilder):
        folders_to_delete = [paths.frame_images_path]
        files_to_delete = [
            paths.image_0_path,
            paths.image_1_path,
            paths.image_2_path,
            paths.image_detection_0_path,
            paths.image_detection_1_path,
            paths.image_detection_2_path,
            paths.result_0_coefficients,
            paths.result_0_image,
            paths.result_0_mesh,
            paths.result_1_coefficients,
            paths.result_1_image,
            paths.result_1_mesh,
            paths.result_2_coefficients,
            paths.result_2_image,
            paths.result_2_mesh,
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

    def _calc_avg_am(self, landmarks_0: np.array, landmarks_1: np.array, landmarks_2: np.array, am: AM):
        return (self._calc_am(landmarks_0, am) + self._calc_am(landmarks_1, am) + self._calc_am(landmarks_2, am)) / 3

    def _calc_am(self, landmarks: np.array, am: AM) -> float:
        if am.type == "distance":
            return self._get_distance(landmarks[am.lm_indicies])
        elif am.type == "distance":
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
