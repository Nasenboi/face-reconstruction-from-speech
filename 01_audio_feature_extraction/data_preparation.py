import os
import sys

sys.path.append("..")
import signal
import subprocess
import threading
from typing import Optional, Union

import audeer
import audonnx
import librosa
import numpy as np
import opensmile
import pandas as pd
import torch
from joblib import Parallel, delayed
from speechbrain.inference.speaker import EncoderClassifier
from tqdm import tqdm
from voice_height_regressor import HeightRegressionPipeline

from src.config import *
from src.models import DataSetRecord
from src.paths import PathBuilder

# Preset Variables if needed:
RANDOM_STATE = 1
np.random.seed(RANDOM_STATE)


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException()


# Preload the model
if FEATURE_SET == "covariants":
    # https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender
    AUDEERING_MODEL = audonnx.load(AUDEER_MODEL_ROOT)

    # https://github.com/griko/voice-height-regression
    HEIGHT_MODEL = HeightRegressionPipeline.from_pretrained(
        "griko/height_reg_svr_ecapa_voxceleb",
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
elif FEATURE_SET == "embeddings":
    EMBEDDING_MODEL = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cuda"})

    def load_audio_librosa(path, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), sr=16000):
        y, sr = librosa.load(path, sr=sr, mono=True)
        signal = torch.from_numpy(y).float().unsqueeze(0).to(device)
        return signal, sr

class DataPreparation:
    def __init__(self):
        self.df = pd.read_feather(DATASET_PATH)

    def calculate_audio_features(
        self, n_jobs: int = 10, drop_na: bool = True, new_path: Optional[str] = None, timeout: int = 60
    ):
        if FEATURE_SET in ["covariants", "embeddings"]:
            n_jobs = 1

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
        if FEATURE_SET in feature_set_map.keys():
            self.smile = opensmile.Smile(feature_set=FEATURE_SET, feature_level=FEATURE_LEVEL)
        elif FEATURE_SET == "mel":
            self.feature_names = [
                f"mel_{i:03d}_{j:03d}"
                for j in range(128)
                for i in range(128)
            ]
        elif FEATURE_SET == "embeddings":
            self.feature_names = [
                f"emb_{i:03d}"
                for i in range(192)
            ]
        
        self.paths = PathBuilder(record)
        self.index = index


    def get_audio_features(
        self,
    ) -> pd.DataFrame:
        try:
            self._extract_audio_from_video()

            if FEATURE_SET in feature_set_map.keys():
                df = self.smile.process_file(self.paths.audio_path)
                df = df.reset_index(drop=True)
            elif FEATURE_SET == "covariants":
                df = self._get_audeer()
            elif FEATURE_SET == "mel":
                df = self._get_mel()
            elif FEATURE_SET == "embeddings":
                df = self._get_embeddings()
            else:
                df = pd.DataFrame([])
                
            df.index = [self.index]
            os.remove(self.paths.audio_path)
            return df
        except Exception as e:
            if os.path.exists(self.paths.audio_path):
                os.remove(self.paths.audio_path)
            raise e
        
    def _get_embeddings(self) -> pd.DataFrame:
        y, fs = load_audio_librosa(self.paths.audio_path)
        embeddings = EMBEDDING_MODEL.encode_batch(y).cpu().numpy()[0][0].reshape(1, -1)

        return pd.DataFrame(
            embeddings,
            columns=self.feature_names
        )
        

    def _get_audeer(self) -> pd.DataFrame:
        y, sr = librosa.load(self.paths.audio_path, mono=True)
        y, _ = librosa.effects.trim(y)

        audeering_model_output: dict = AUDEERING_MODEL(y, sr)
        height_model_output: dict = HEIGHT_MODEL(self.paths.audio_path)[0]

        return pd.DataFrame([
            {
                "esitmated_age": audeering_model_output["logits_age"][0],
                "estimated_gender_m": audeering_model_output["logits_gender"][0][0],
                "estimated_gender_f": audeering_model_output["logits_gender"][0][1],
                "estimated_gender_c": audeering_model_output["logits_gender"][0][2],
                "estimated_height": height_model_output
            }
        ])



    def _get_mel(self) -> pd.DataFrame:
        y, sr = librosa.load(self.paths.audio_path, mono=True)
        y, _ = librosa.effects.trim(y)
        
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=F_MAX)
        
        if S.shape[1] > 128:
            S_trimmed = S[:, :128]
        elif S.shape[1] < 128:
            padding = 128 - S.shape[1]
            S_trimmed = np.pad(S, ((0, 0), (0, padding)), mode="constant")
        else:
            S_trimmed = S
        if S_trimmed.shape[0] != 128:
            if S_trimmed.shape[0] > 128:
                S_trimmed = S_trimmed[:128, :]
            else:
                padding = 128 - S_trimmed.shape[0]
                S_trimmed = np.pad(S_trimmed, ((0, padding), (0, 0)), mode="constant")
        
        flattened_features = S_trimmed.flatten()

        df = pd.DataFrame([flattened_features], columns=self.feature_names)
        
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
                self.paths.audio_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if not os.path.exists(self.paths.video_path):
            raise FileNotFoundError
