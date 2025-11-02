#!/bin/bash

usage() {
  echo "Usage: $0 -i IMAGE_FOLDER -m MODEL_NAME -e EPOCH"
  exit 1
}

# parse options
while getopts "i:m:e:h" opt; do
  case "$opt" in
    i) IMAGE_FOLDER="$OPTARG" ;;
    m) MODEL_NAME="$OPTARG" ;;
    e) EPOCH="$OPTARG" ;;
    h|*) usage ;;
  esac
done

# validate required args
if [ -z "$IMAGE_FOLDER" ] || [ -z "$MODEL_NAME" ] || [ -z "$EPOCH" ]; then
  usage
fi

export PATH="/root/miniconda3/envs/deep3d_pytorch/bin:$PATH"

cd /app/Deep3DFaceRecon_pytorch


/root/miniconda3/envs/deep3d_pytorch/bin/python /app/Deep3DFaceRecon_pytorch/test.py --name="$MODEL_NAME" --epoch="$EPOCH" --img_folder="$IMAGE_FOLDER"

# python /app/Deep3DFaceRecon_pytorch/data_preparation.py --img_folder="$IMAGE_FOLDER"