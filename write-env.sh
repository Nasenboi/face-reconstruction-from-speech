#! /bin/bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config="$script_dir/config.json"


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

touch .env

# Create or overwrite .env file
cat > .env << EOF
# Auto-generated from config.json
IMAGE_FOLDER=$(jq -r '.paths.image' "$config")
VIDEO_FOLDER=$(jq -r '.paths.video' "$config")
MESH_FOLDER=$(jq -r '.paths.mesh' "$config")
BFM_FOLDER=$(jq -r '.paths.bfm' "$config")
CHECKPOINT_FOLDER=$(jq -r '.paths.checkpoints' "$config")
MODEL_NAME=$(jq -r '.face_geometry.model' "$config")
EPOCH=$(jq -r '.face_geometry.epoch' "$config")
API_PORT=$(jq -r '.face_geometry.api_port' "$config")
DATASETS_PATH=$(jq -r '.paths.datasets' "$config")
DATASET_NAME=$(jq -r '.dataset.name' "$config")
STEP_SIZE=$(jq -r '.face_geometry.frame_step_size' "$config")
BUFFER_SIZE=$(jq -r '.face_geometry.frame_buffer_size' "$config")
TIMEOUT=$(jq -r '.face_geometry.timeout' "$config")
EOF

echo ".env file created successfully!"
