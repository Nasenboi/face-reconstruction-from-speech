#!/bin/bash

ENV_NAME="acoustic-to-anthropometric"
if ! which python | grep -q "$ENV_NAME"; then
    echo "Error: Are you sure this is the right conda environment?"
    echo "Current Python path: $(which python)"
    exit 1
fi

root=$(pwd)

cd "$root/01_dataset_preparation" && python main.py || exit 1

cd "$root"
sh "02_face_landmark_detection/run.sh" -d || exit 1
sh "03_face_geometry_estimation/run.sh" || exit 1

cd "$root/04_am_calculation" && python main.py || exit 1