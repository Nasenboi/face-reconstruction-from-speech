import json
import os

from voxceleb.consts import COOKIE_PATH, TEST_DATA_PATH, TRAIN_DATA_PATH
from voxceleb.data_loader import VoxCelebDataLoader

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(parent_dir, "config.json")
config = json.load(open(config_path))


data_loader = VoxCelebDataLoader(
    video_path=config["paths"]["video"],
    audio_path=config["paths"]["audio"],
    image_path=config["paths"]["image"],
    datasets_path=config["paths"]["datasets"],
    # train_data_path=TRAIN_DATA_PATH,
    test_data_path=TEST_DATA_PATH,
    cookie_path=COOKIE_PATH,
    max_records=100,
)
