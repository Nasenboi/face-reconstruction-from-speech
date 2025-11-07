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
        self.frame_images_path = os.path.join(CONFIG["paths"]["frames"], dataset_id)
        self.image_0_path = os.path.join(CONFIG["paths"]["image"], f"{dataset_id}_0.jpg")
        self.image_1_path = os.path.join(CONFIG["paths"]["image"], f"{dataset_id}_1.jpg")
        self.image_2_path = os.path.join(CONFIG["paths"]["image"], f"{dataset_id}_2.jpg")
        self.result_0_image = os.path.join(results_dir, f"{dataset_id}_0.png")
        self.result_1_image = os.path.join(results_dir, f"{dataset_id}_1.png")
        self.result_2_image = os.path.join(results_dir, f"{dataset_id}_2.png")
        self.result_0_mesh = os.path.join(results_dir, f"{dataset_id}_0.obj")
        self.result_1_mesh = os.path.join(results_dir, f"{dataset_id}_1.obj")
        self.result_2_mesh = os.path.join(results_dir, f"{dataset_id}_2.obj")
        self.result_0_coefficients = os.path.join(results_dir, f"{dataset_id}_0.mat")
        self.result_1_coefficients = os.path.join(results_dir, f"{dataset_id}_1.mat")
        self.result_2_coefficients = os.path.join(results_dir, f"{dataset_id}_2.mat")
