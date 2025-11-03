import json
import os

from voxceleb.consts import ARGS
from voxceleb.data_loader import VoxCelebDataLoader

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(parent_dir, "config.json")
config = json.load(open(config_path))


data_loader = VoxCelebDataLoader(
    video_path=config["paths"]["video"],
    audio_path=config["paths"]["audio"],
    image_path=config["paths"]["image"],
    datasets_path=config["paths"]["datasets"],
    max_records=100,
    **ARGS
)
