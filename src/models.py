from typing import Literal

from pydantic import BaseModel


class DataSetRecord(BaseModel):
    speaker_id: str
    face_id: str
    video_id: str
    clip_id: str
    gender: Literal["m", "f"]
    split: Literal["test", "train"]
    batch: int
