import json
import os

from .config import *
from .models import DataSetRecord


class PathBuilder:
    """
    The path builder class.
    This class generates the paths to files based on the dataset ID.
    """

    def __init__(self, record: DataSetRecord):
        """
        Initializes the path builder with the given dataset ID.
        :param dataset_id: The dataset ID.
        """
        results_dir = os.path.join(
            CONFIG["paths"]["checkpoints"],
            CONFIG["face_geometry"]["model"],
            "results",
            f"epoch_{CONFIG['face_geometry']['epoch']}_000000",
        )

        dataset_id = f"{record.speaker_id}_{record.video_id}_{record.clip_id}"
        self.dataset_id = dataset_id

        self.video_path = os.path.join(
            CONFIG["paths"]["video"], record.speaker_id, record.video_id, f"{record.clip_id}.mp4"
        )
