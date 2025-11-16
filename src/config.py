import json
import os
from typing import List

from opensmile import FeatureLevel, FeatureSet

from .models import AM, feature_level_map, feature_set_map

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
FEATURE_LEVEL: FeatureLevel = feature_level_map[CONFIG["dataset"]["feature_level"]]

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lm_map_path = os.path.join(parent_dir, "am_map.json")
AM_MAP: dict = json.load(open(lm_map_path))

dist_ams: List[AM] = [AM(type="distance", lm_indicies=am) for am in AM_MAP["distance"]]
angle_ams: List[AM] = [AM(type="angle", lm_indicies=am) for am in AM_MAP["angle"]]
prop_ams: List[AM] = [AM(type="proportion", lm_indicies=am) for am in AM_MAP["proportion"]]
AMS: List[AM] = dist_ams + angle_ams + prop_ams

AM_COLUMN_NAMES: List[str] = [am.get_column_name() for am in AMS]


# Create paths if they dont exist
[os.makedirs(p, exist_ok=True) for p in [VIDEO_PATH, TMP_AUDIO_PATH, IMAGE_PATH]]
