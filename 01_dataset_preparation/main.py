from argparse import ArgumentParser

from voxceleb.consts import ARGS
from voxceleb.data_loader import VoxCelebDataLoader

parser = ArgumentParser(description="Prepare the dataset.")
parser.add_argument("-m", "--max_records", help="Set the maximum of database records to prepare")

args = parser.parse_args()

data_loader = VoxCelebDataLoader(max_records=int(args.max_records), **ARGS)
