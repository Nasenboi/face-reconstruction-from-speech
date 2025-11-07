import os
import shutil
from argparse import ArgumentParser
from math import sqrt

import numpy as np
import tensorflow as tf
import tqdm
from mtcnn import MTCNN
from mtcnn.utils.images import load_image

device = "gpu:0" if len(tf.config.list_physical_devices("GPU")) > 0 else "cpu"

# Create a detector instance
detector = MTCNN(device=device)


def process_image(image_path: str):
    image_name = os.path.basename(image_path).split(".")[0]
    output_path = os.path.join(txt_folder, f"{image_name}.txt")

    if os.path.exists(output_path):
        return

    image = load_image(image_path)
    result = detector.detect_faces(image)
    points = result[0]["keypoints"]
    box = result[0]["box"]
    return points, box


def calculate_forward_degree(keypoints: dict, box: list) -> float:
    box_x_center = box[0] + round(box[2] / 2)
    eye_x_center = (keypoints["left_eye"][0] + keypoints["right_eye"][0]) / 2
    mouth_x_center = (keypoints["mouth_left"][0] + keypoints["mouth_right"][0]) / 2

    # eye_center_distance = abs(box_x_center - eye_x_center) / box[2]
    # mouth_center_distance = abs(box_x_center - mouth_x_center) / box[2]
    # nose_center_distance = abs(box_x_center - keypoints["nose"][0]) / box[2]
    # center_distances = (eye_center_distance + mouth_center_distance + nose_center_distance) / 3

    eye_center_nose_distance = abs(eye_x_center - keypoints["nose"][0]) / box[2]
    mouth_center_nose_distance = abs(mouth_x_center - keypoints["nose"][0]) / box[2]

    return (eye_center_nose_distance + mouth_center_nose_distance) / 2


parser = ArgumentParser(description="Detect faces in all images in a folder.")
parser.add_argument("-f", "--frame_folder", required=True, help="Path to folder containing frames")
parser.add_argument("-i", "--img_folder", required=True, help="Path to folder containing images")
parser.add_argument("-d", "--delete", action="store_true", help="Choose to delete the frame folder or not")

args = parser.parse_args()
img_folder: str = args.img_folder
frame_folder: str = args.frame_folder
delete_frame_folder: bool = args.delete

txt_folder = os.path.join(img_folder, "detections")
os.makedirs(txt_folder, exist_ok=True)

exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
frame_folders = os.listdir(frame_folder)

frame_folder_iterator = tqdm.tqdm(frame_folders, desc="Processing frame folders")
for dataset_id in frame_folder_iterator:
    frames_path = os.path.join(frame_folder, dataset_id)
    frames = [f for f in os.listdir(frames_path) if f.lower().endswith(exts)]

    frame_data = []

    for f in frames:
        frame_path = os.path.join(frames_path, f)

        try:
            keypoints, box = process_image(frame_path)
            pseudo_angle = calculate_forward_degree(keypoints, box)
            frame_data.append({"frame_path": frame_path, "keypoints": keypoints, "pseudo_angle": pseudo_angle})
        except Exception as e:
            print(f"Error processing frame {dataset_id}, {f}:\n{e}")
            continue

    best_frames = sorted(frame_data, key=lambda x: x["pseudo_angle"])[:3]

    for i, bf in enumerate(best_frames):
        file_base_name = f"{dataset_id}_{i}"

        copy_path = os.path.join(img_folder, f"{file_base_name}.jpg")
        if not os.path.exists(copy_path):
            shutil.copyfile(bf["frame_path"], copy_path)

        txt_path = os.path.join(txt_folder, f"{file_base_name}.txt")
        if not os.path.exists(txt_path):
            with open(txt_path, "w") as f:
                for key, point in bf["keypoints"].items():
                    f.write(f"{point[0]} {point[1]}\n")

    if delete_frame_folder:
        shutil.rmtree(frames_path)
