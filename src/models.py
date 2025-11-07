from typing import List, Literal

from opensmile import FeatureSet
from pydantic import BaseModel, field_validator


class AM(BaseModel):
    lm_indicies: List[int]
    type: Literal["distance", "angle", "proportion"]


class DataSetRecord(BaseModel):
    speaker_id: str
    face_id: str
    video_id: str
    clip_id: str
    gender: Literal["m", "f"]
    split: Literal["test", "train"]
    batch: int


feature_set_map = {"ComParE_2016": FeatureSet.ComParE_2016, "eGeMAPSv02": FeatureSet.eGeMAPSv02}
