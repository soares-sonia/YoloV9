#!/bin/bash

# Step 1: Create the virtual environment
echo "Creating virtual environment..."
python -m venv yolov9_env

# Step 2: Activate the virtual environment
echo "Activating virtual environment..."
source yolov9_env/bin/activate

# Step 3: Install dependencies
echo "Installing dependencies..."
pip install torch torchvision torchaudio
pip install opencv-python
pip install matplotlib
pip install pandas
pip install seaborn
pip install roboflow

# Step 4: Clone YOLOv9 repository
echo "Cloning YOLOv9 repository..."
git clone https://github.com/WongKinYiu/yolov9.git
cd yolov9

# Step 5: Install repository-specific dependencies
echo "Installing repository-specific dependencies..."
pip install -r requirements.txt

# Step 6: Download YOLOv9 weights
echo "Downloading YOLOv9 weights..."
mkdir -p weights
wget https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt -P weights
wget https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt -P weights
wget https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c.pt -P weights
wget https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-e.pt -P weights 

# Notify user that setup is complete
echo "Setup complete. You can now run the YOLOv9 model using the detect.py script."

# Example usage:
# python detect.py --source data/images/horses.jpg --weights weights/gelan-c.pt --conf 0.25 --name yolov9_results