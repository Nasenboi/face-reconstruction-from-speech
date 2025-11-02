import os
from argparse import ArgumentParser

import tensorflow as tf
import tqdm
from mtcnn import MTCNN
from mtcnn.utils.images import load_image

device = "gpu:0" if len(tf.config.list_physical_devices("GPU")) > 0 else "cpu"

# Create a detector instance
detector = MTCNN(device=device)


def process_image(image_path: str, txt_folder: str):
    image_name = os.path.basename(image_path).split(".")[0]
    output_path = os.path.join(txt_folder, f"{image_name}.txt")

    if os.path.exists(output_path):
        return

    image = load_image(image_path)
    result = detector.detect_faces(image)
    points = result[0]["keypoints"]

    with open(output_path, "w") as f:
        for key, point in points.items():
            f.write(f"{point[0]} {point[1]}\n")


parser = ArgumentParser(description="Detect faces in all images in a folder.")
parser.add_argument("-i", "--img_folder", required=True, help="Path to folder containing images")

args = parser.parse_args()
img_folder = args.img_folder
txt_folder = os.path.join(img_folder, "detections")

if not os.path.exists(txt_folder):
    os.mkdir(txt_folder)

exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
files = [f for f in os.listdir(img_folder) if f.lower().endswith(exts)]


iterator = tqdm.tqdm(sorted(files), desc="Processing images")
for fname in iterator:
    img_path = os.path.join(img_folder, fname)
    try:
        res = process_image(img_path, txt_folder)
        # print(f'{img_path}: {res}')
    except Exception as e:
        print(f"Failed processing {img_path}: {e}")
