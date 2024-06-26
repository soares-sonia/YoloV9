#!/bin/bash

# Check if model path and capture mode are provided
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: ./yolov9/run_detection.sh <path_to_model> <screen/webcam>"
  exit 1
fi

MODEL_PATH=$1
DETECT=$2

# Activate the virtual environment
echo "Activating virtual environment..."
source yolov9_env/bin/activate

# Run the screen capture detection script with the model path and capture mode
echo "Running screen capture detection..."
python yolov9/real_time_detection.py $MODEL_PATH $DETECT
