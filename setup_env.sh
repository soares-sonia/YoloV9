#!/bin/bash

# Step 1: Create the virtual environment
echo "Creating virtual environment..."
python3 -m venv yolov9_env

# Step 2: Activate the virtual environment
echo "Activating virtual environment..."
source yolov9_env/bin/activate

# Step 3: Install system dependencies
sudo apt update
sudo apt install libgtk2.0-dev pkg-config
sudo apt install libgl1-mesa-glx

# Step 4: Install dependencies
echo "Installing dependencies..."
pip install torch torchvision torchaudio
pip install numpy
pip install opencv-python
pip install matplotlib
pip install pandas
pip install seaborn
pip install roboflow
pip install mss
pip install ipython
pip install psutil

# Step 5: Clone YOLOv9 repository
echo "Cloning YOLOv9 repository..."
git clone https://github.com/WongKinYiu/yolov9.git
cd yolov9

# Step 6: Install repository-specific dependencies
echo "Installing repository-specific dependencies..."
pip install -r requirements.txt

# Step 7: Download YOLOv9 weights
echo "Downloading YOLOv9 weights..."
mkdir -p weights
wget https://github.com/soares-sonia/YoloV9/releases/download/v0.1/best.pt -P weights
wget https://github.com/soares-sonia/YoloV9/releases/download/v0.1/best_striped.pt -P weights
wget https://github.com/soares-sonia/YoloV9/releases/download/v0.1/last.pt -P weights
wget https://github.com/soares-sonia/YoloV9/releases/download/v0.1/last_striped.pt -P weights 

# Notify user that setup is complete
echo "Setup complete. You can now run the YOLOv9 model using the run_detection.sh script."

# Example usage:
# python detect.py --source data/images/horses.jpg --weights weights/gelan-c.pt --conf 0.25 --name yolov9_results