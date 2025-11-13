from typing import List, Literal

from opensmile import FeatureLevel, FeatureSet
from pydantic import BaseModel


class AM(BaseModel):
    lm_indicies: List[int]
    type: Literal["distance", "angle", "proportion"]

    def get_column_name(self):
        return f"{self.type}_" + "_".join([f"{i:02d}" for i in self.lm_indicies])


class DataSetRecord(BaseModel):
    speaker_id: str
    face_id: str
    video_id: str
    clip_id: str
    gender: Literal["m", "f"]
    split: Literal["test", "train"]
    batch: int


feature_set_map = {"ComParE_2016": FeatureSet.ComParE_2016, "eGeMAPSv02": FeatureSet.eGeMAPSv02}
feature_level_map = {
    "Functionals": FeatureLevel.Functionals,
    "LowLevelDescriptors": FeatureLevel.LowLevelDescriptors,
    "LowLevelDescriptors_Deltas": FeatureLevel.LowLevelDescriptors_Deltas,
}
