#!/bin/bash

# locate this script and the parent config.json
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config="$script_dir/../config.json"


# require jq (install on Debian/Ubuntu: sudo apt install jq)
if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required. Install: sudo apt install jq" >&2
  exit 1
fi

# check if config file exists
if [ ! -f "$config" ]; then
  echo "Config file not found: $config" >&2
  exit 1
fi

image_folder=$(jq -r '.paths.image' "$config")
mesh_folder=$(jq -r '.paths.mesh' "$config")
bfm_folder=$(jq -r '.paths.bfm' "$config")
checkpoint_folder=$(jq -r '.paths.checkpoints' "$config")
model_name=$(jq -r '.face_geometry.model' "$config")
epoch=$(jq -r '.face_geometry.epoch' "$config")
api_port=$(jq -r '.face_geometry.api_port' "$config")

docker build -t geometry_estimator:latest $script_dir && \
docker run --rm -it \
-v "$image_folder":/app/input \
-v "$mesh_folder":/app/mesh \
-v "$bfm_folder":/app/Deep3DFaceRecon_pytorch/BFM \
-v "$checkpoint_folder":/app/Deep3DFaceRecon_pytorch/checkpoints \
--gpus all \
-p 8000:8000 \
geometry_estimator:latest \
-i /app/input/ \
-o /app/mesh/ \
-m "$model_name" \
-e "$epoch"