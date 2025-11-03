import json
import os

# Load Config
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(parent_dir, "config.json")
config = json.load(open(config_path))


TEST_DATASET_PATH = os.path.join(config["paths"]["datasets"], "test.csv")
TRAIN_DATASET_PATH = os.path.join(config["paths"]["datasets"], "train.csv")
VIDEO_PATH = config["paths"]["video"]
AUDIO_PATH = config["paths"]["audio"]
IMAGE_PATH = config["paths"]["image"]
DATASET_PATH = config["paths"]["datasets"]
BFM_PATH = os.path.join(config["paths"]["bfm"])


class PathBuilder:
    """
    The path builder class.
    This class generates the paths to files based on the dataset ID.
    """

    def __init__(self, dataset_id: str):
        """
        Initializes the path builder with the given dataset ID.
        :param dataset_id: The dataset ID.
        """
        results_dir = os.path.join(
            config["paths"]["checkpoints"],
            config["face_geometry"]["model"],
            "results",
            f"epoch_{config['face_geometry']['epoch']}_000000",
        )

        self.dataset_id = dataset_id
        self.video_path = os.path.join(config["paths"]["video"], f"{dataset_id}.mp4")
        self.audio_path = os.path.join(config["paths"]["audio"], f"{dataset_id}.wav")
        self.image_S_path = os.path.join(config["paths"]["image"], f"{dataset_id}_S.jpg")
        self.image_M_path = os.path.join(config["paths"]["image"], f"{dataset_id}_M.jpg")
        self.image_E_path = os.path.join(config["paths"]["image"], f"{dataset_id}_E.jpg")
        self.result_S_image = os.path.join(results_dir, f"{dataset_id}_S.png")
        self.result_M_image = os.path.join(results_dir, f"{dataset_id}_M.png")
        self.result_E_image = os.path.join(results_dir, f"{dataset_id}_E.png")
        self.result_S_mesh = os.path.join(results_dir, f"{dataset_id}_S.obj")
        self.result_M_mesh = os.path.join(results_dir, f"{dataset_id}_M.obj")
        self.result_E_mesh = os.path.join(results_dir, f"{dataset_id}_E.obj")
        self.result_S_coefficients = os.path.join(results_dir, f"{dataset_id}_S.mat")
        self.result_M_coefficients = os.path.join(results_dir, f"{dataset_id}_M.mat")
        self.result_E_coefficients = os.path.join(results_dir, f"{dataset_id}_E.mat")
