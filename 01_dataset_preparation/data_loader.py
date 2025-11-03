from typing import Optional


class DataLoader:
    """
    Data loader example class.
    There is no need of inheriting this class,
    it just serves as an example of what functions
    and parameters a data loader should have.
    """

    def __init__(
        self,
        video_path: str,
        audio_path: str,
        image_path: str,
        datasets_path: str,
        max_records: Optional[int] = None,
    ):
        self.video_path = video_path
        self.audio_path = audio_path
        self.image_path = image_path
        self.datasets_path = datasets_path
        self.max_records = max_records

        self.read_data()
        self.create_dataset()

    def read_data(self):
        """Reads the data from the chosen dataset and stores it internally."""
        pass

    def create_dataset(self):
        """
        Creates the dataset from the read data.
        This should fill at least two directories: image_path and audio_path.
        Additionally the test and train datasets csv files should be created here as well.
        """
        pass
