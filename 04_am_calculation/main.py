from am_calculator import AMCalculator

from src.config import *

am_calculator = AMCalculator(DATASET_PATH)
am_calculator.calculate_ams(delete_interm_files=True)
