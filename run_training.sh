#!/bin/bash

# Variables (update these paths according to your setup)
YOLOV9_DIR="yolov9"
DATASET_LOCATION="BANANA.v1i.yolov9/data.yaml"
WEIGHTS_PATH="weights/gelan-c.pt"
CFG_PATH="models/detect/gelan-c.yaml"
HYP_PATH="hyp.scratch-high.yaml"

# Navigate to the YOLOv9 directory
echo "Navigating to YOLOv9 directory..."
cd $YOLOV9_DIR

# Run the training command
echo "Running the training script..."
python train.py \
  --batch 16 \
  --epochs 25 \
  --img 640 \
  --device 0 \
  --min-items 0 \
  --close-mosaic 15 \
  --data $DATASET_LOCATION \
  --weights $WEIGHTS_PATH \
  --cfg $CFG_PATH \
  --hyp $HYP_PATH

echo "Training script finished."
