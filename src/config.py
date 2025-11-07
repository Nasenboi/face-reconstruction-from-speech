import json
import os

from opensmile import FeatureSet

from .models import feature_set_map

# Load Config
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(parent_dir, "config.json")
CONFIG: dict = json.load(open(config_path))


DATASET_PATH: str = os.path.join(CONFIG["paths"]["datasets"], CONFIG["dataset"]["name"])
VIDEO_PATH: str = CONFIG["paths"]["video"]
TMP_AUDIO_PATH: str = CONFIG["paths"]["tmp_audio"]
IMAGE_PATH: str = CONFIG["paths"]["image"]
BFM_PATH: str = os.path.join(CONFIG["paths"]["bfm"])

FEATURE_SET: FeatureSet = feature_set_map[CONFIG["dataset"]["feature_set"]]
