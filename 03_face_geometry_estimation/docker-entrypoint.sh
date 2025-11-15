#!/bin/bash

usage() {
  echo "Usage: $0 -i IMAGE_FOLDER -m MODEL_NAME -e EPOCH -o MESH_FOLDER"
  exit 1
}

# parse options
while getopts "i:m:e:o:h" opt; do
  case "$opt" in
    i) IMAGE_FOLDER="$OPTARG" ;;
    m) MODEL_NAME="$OPTARG" ;;
    e) EPOCH="$OPTARG" ;;
    o) MESH_FOLDER="$OPTARG" ;;
    h|*) usage ;;
  esac
done

# validate required args
if [ -z "$IMAGE_FOLDER" ] || [ -z "$MODEL_NAME" ] || [ -z "$EPOCH" ] || [ -z "$MESH_FOLDER" ]; then
  usage
fi

export PATH="/root/miniconda3/envs/deep3d_pytorch/bin:$PATH"

cd /app/Deep3DFaceRecon_pytorch

echo "starting main.py"

/root/miniconda3/envs/deep3d_pytorch/bin/python /app/Deep3DFaceRecon_pytorch/main.py --name="$MODEL_NAME" --epoch="$EPOCH" --img_folder="$IMAGE_FOLDER" --mesh_folder="$MESH_FOLDER"

echo "should not land here!"

# python /app/Deep3DFaceRecon_pytorch/data_preparation.py --img_folder="$IMAGE_FOLDER"