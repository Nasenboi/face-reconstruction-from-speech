import os
import signal
from argparse import ArgumentParser
from typing import Literal, Optional

import cv2
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from mtcnn import MTCNN
from mtcnn.utils.images import load_image
from pydantic import BaseModel
from tqdm import tqdm

# Globals
## args
parser = ArgumentParser(description="Detect faces in all images in a folder.")
parser.add_argument("--video_folder", required=True, help="Path to folder containing videos")
parser.add_argument("--img_folder", required=True, help="Path to folder containing images")
parser.add_argument("--dataset_path", required=True, help="Path to the dataset")
parser.add_argument("--buffer_size", required=True, help="Size of the start and end buffer in frames")
parser.add_argument("--step_size", required=True, help="Size of the steos to take while iterating frames")
parser.add_argument("--port", required=True, help="Port of the api endpoint")
parser.add_argument("--host", required=True, help="Host address of the api endpoint")
parser.add_argument("--timeout", required=True, help="Timeout of processes in seconds")
args = parser.parse_args()
VIDEO_FOLDER = args.video_folder
IMAGE_FOLDER = args.img_folder
TXT_FOLDER = os.path.join(IMAGE_FOLDER, "detections")
os.makedirs(TXT_FOLDER, exist_ok=True)
DATASET_PATH = args.dataset_path
BUFFER_SIZE = int(args.buffer_size)
STEP_SIZE = int(args.step_size)
port = args.port
host = args.host
URL = f"http://{host}:{port}/get-landmarks"
TIMEOUT = int(args.timeout)

# ToDo: Add as config vars
DROP_NA = False
MAX_FRAMES = 10
STOP = MAX_FRAMES * STEP_SIZE


##  detector instance
device = "gpu:0" if len(tf.config.list_physical_devices("GPU")) > 0 else "cpu"
DETECTOR = MTCNN(device=device)


# Classes
class DataSetRecord(BaseModel):
    speaker_id: str
    face_id: str
    video_id: str
    clip_id: str
    gender: Literal["m", "f"]
    split: Literal["test", "train"]
    batch: int


class PathBuilder:
    """
    The path builder class.
    This class generates the paths to files based on the record.
    """

    def __init__(self, record: DataSetRecord):
        """
        Initializes the path builder with the given dataset record
        """
        dataset_id = f"{record.speaker_id}_{record.video_id}_{record.clip_id}"
        self.dataset_id = dataset_id

        self.video_path = os.path.join(VIDEO_FOLDER, record.speaker_id, record.video_id, f"{record.clip_id}.mp4")


class TimeoutException(Exception):
    pass


# Functions
def timeout_handler(signum, frame):
    raise TimeoutException()


def calculate_forwardness(keypoints: dict, box: list) -> float:
    # box_x_center = box[0] + round(box[2] / 2)
    eye_x_center = (keypoints["left_eye"][0] + keypoints["right_eye"][0]) / 2
    mouth_x_center = (keypoints["mouth_left"][0] + keypoints["mouth_right"][0]) / 2

    # eye_center_distance = abs(box_x_center - eye_x_center) / box[2]
    # mouth_center_distance = abs(box_x_center - mouth_x_center) / box[2]
    # nose_center_distance = abs(box_x_center - keypoints["nose"][0]) / box[2]
    # center_distances = (eye_center_distance + mouth_center_distance + nose_center_distance) / 3

    eye_center_nose_distance = abs(eye_x_center - keypoints["nose"][0]) / box[2]
    mouth_center_nose_distance = abs(mouth_x_center - keypoints["nose"][0]) / box[2]

    return (eye_center_nose_distance + mouth_center_nose_distance) / 2


def process_image(image_path: str) -> dict:
    image = load_image(image_path)
    result = DETECTOR.detect_faces(image)
    points = result[0]["keypoints"]
    box = result[0]["box"]
    forwardness = calculate_forwardness(points, box)
    return {"keypoints": points, "forwardness": forwardness}


def process_frame(paths, cap, frame_no):
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret:
            return None

        frame_path = os.path.join(IMAGE_FOLDER, f"{paths.dataset_id}_{frame_no}.jpg")
        cv2.imwrite(frame_path, frame)

        if not os.path.exists(frame_path):
            return None

        return {"frame_path": frame_path, **process_image(frame_path)}
    except Exception as e:
        return None


def write_txt_file(frame: dict):
    txt_path = os.path.join(TXT_FOLDER, os.path.basename(frame["frame_path"]).replace(".jpg", ".txt"))
    with open(txt_path, "w") as f:
        for key, point in frame["keypoints"].items():
            f.write(f"{point[0]} {point[1]}\n")


def send_request(filename: str) -> np.array:
    response = requests.post(URL, params={"filename": filename})
    data = response.json()
    if response.status_code != 200:
        print(f"Error sending request:\n{data}")
        response.raise_for_status()

    landmarks_list = data["landmarks"]
    return np.array(landmarks_list)


def remove_files(dataset_id: str):
    img_files = os.listdir(IMAGE_FOLDER)
    txt_files = os.listdir(TXT_FOLDER)
    [os.remove(os.path.join(IMAGE_FOLDER, i)) for i in img_files if dataset_id in i]
    [os.remove(os.path.join(TXT_FOLDER, t)) for t in txt_files if dataset_id in t]


def create_landmark_dict(landmarks: np.array):
    lm_dict = {}
    for i, lm in enumerate(landmarks):
        for j, p in enumerate(lm):
            lm_dict[f"landmark_{i:02d}_{j}"] = p
    return lm_dict


def process_video(index, record: DataSetRecord) -> pd.DataFrame:
    paths = PathBuilder(record)
    dataset_id = paths.dataset_id
    frames = []
    df = None
    try:
        cap = cv2.VideoCapture(paths.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {paths.video_path}")

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_indices = range(BUFFER_SIZE, min(STOP, num_frames - BUFFER_SIZE), STEP_SIZE)
        frames = [process_frame(paths, cap, f) for f in frame_indices]
        frames = [f for f in frames if f is not None]

        top_3_frames = sorted(frames, key=lambda x: x["forwardness"])[:3]

        landmarks = []
        for f in top_3_frames:
            write_txt_file(f)
            landmarks.append(send_request(os.path.basename(f["frame_path"])))
        landmarks = np.array(landmarks)
        avg_landmarks = np.mean(landmarks, axis=0)
        landmark_dict = create_landmark_dict(avg_landmarks)
        df = pd.DataFrame({**landmark_dict}, index=[index])
    except Exception as e:
        print(f"Could not process video {paths.video_path}:\n{e}")
    remove_files(dataset_id)
    return df


def iterate_df(df: pd.DataFrame, timeout=900, new_path: Optional[str] = None):
    def get_facial_landmarks(
        record_tuple,
    ):
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            index, record_dict = record_tuple
            record = DataSetRecord(**record_dict)
            return process_video(index, record)
        except TimeoutException:
            print(f"Timeout processing record for {index} {record.clip_id}!")
            return None
        except Exception as e:
            print(f"Error calculating audio features for {index} {record.clip_id}:\n{e}")
            return None

    records_to_process = [(index, record.to_dict()) for index, record in df.iterrows()]

    all_landmarks = [
        get_facial_landmarks(record_data)
        for record_data in tqdm(records_to_process, desc="Calculating Facial Landmarks")
    ]

    if DROP_NA:
        print("Dropping Na:")
        all_landmarks = [f for f in all_landmarks if f is not None]

    print("Concatinating Dicts")
    audio_feature_df = pd.concat(all_landmarks, axis=0)
    df = df.merge(audio_feature_df, left_index=True, right_index=True, how="left")

    print("Resetting Index")
    df = df.reset_index(drop=True)

    print("Saving to Feather...")
    output_path = new_path if new_path is not None else DATASET_PATH
    df.to_feather(output_path, compression="zstd", compression_level=3)


data = pd.read_feather(DATASET_PATH)
if DROP_NA:
    data = data.dropna()
new_path = DATASET_PATH.replace(".feather", "_lms.feather")
iterate_df(data, timeout=TIMEOUT, new_path=new_path)
