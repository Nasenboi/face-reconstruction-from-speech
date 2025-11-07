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
frames_folder=$(jq -r '.paths.frames' "$config")

docker build -t landmark_detector:latest $script_dir && \
docker run --rm -it \
-v "$frames_folder":/app/frames \
-v "$image_folder":/app/image \
--gpus all \
landmark_detector:latest \
-i /app/image \
-f /app/frames \
"$@"