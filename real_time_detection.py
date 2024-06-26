import cv2
import numpy as np
from mss import mss
from torchvision import transforms
from yolov9.models.common import DetectMultiBackend
from yolov9.utils.general import non_max_suppression
from yolov9.utils.torch_utils import select_device
import sys
import time

# Check if model path is provided as a command-line argument
if len(sys.argv) < 3:
    print("Usage: python screen_webcam_detection.py <path_to_model> <screen/webcam>")
    sys.exit(1)

# Path to your model
WEIGHTS_PATH = sys.argv[1]
DETECT = sys.argv[2]
DEVICE = "cpu"  # Use "cuda" if you have a GPU

# Load the YOLOv9 model
device = select_device(DEVICE)
model = DetectMultiBackend(WEIGHTS_PATH, device=device, dnn=False)
stride, names, pt = model.stride, model.names, model.pt
img_size = 640

# Transformation
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Initialize screen capture
sct = mss()

# Screen capture parameters
monitor = sct.monitors[1]  # Adjust this if you have multiple monitors

# Custom scale_coords function
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].clip(min=0, max=max(img0_shape))  # clip coordinates
    return coords

def detect_objects(image):
    # Resize image to match the model's input size
    img = cv2.resize(image, (img_size, img_size))
    img = transform(img).to(device)
    img = img.unsqueeze(0)
    pred = model(img, augment=False)
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)
    return pred, img.shape[2:]  # Return prediction and resized image shape

# Function to run object detection from screen capture
def detect_objects_screen_capture():
    # Capture screenshot
    screenshot = np.array(sct.grab(monitor))
    screenshot_rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2RGB)
    
    # Run detection
    pred, img_shape = detect_objects(screenshot_rgb)
    
    # Draw bounding boxes
    result_image = draw_boxes(screenshot_rgb, pred, img_shape)
    
    return result_image

# Function to run object detection from webcam
def detect_objects_webcam(cap):
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam.")
        return None
    
    # Run detection
    pred, img_shape = detect_objects(frame)
    
    # Draw bounding boxes
    result_image = draw_boxes(frame, pred, img_shape)
    
    return result_image

# Function to draw bounding boxes
def draw_boxes(image, pred, img_shape):
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img_shape, det[:, :4], image.shape).round()
            for *xyxy, conf, cls in det:
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, image, label=label, color=(255, 0, 0), line_thickness=2)
    return image

# Function to plot bounding boxes
def plot_one_box(xyxy, img, color=(128, 128, 128), label=None, line_thickness=3):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

# Frame rate limiter
frame_rate = 30  # Adjust as needed
prev_time = 0

# Open webcam capture
cap = cv2.VideoCapture(0)  # Use 0 for default webcam, adjust if you have multiple cameras

while True:
    # Limit the frame rate
    current_time = time.time()
    if current_time - prev_time < 1.0 / frame_rate:
        continue
    prev_time = current_time

    # Use screen capture
    if DETECT == "screen":
        result_image = detect_objects_screen_capture()
    elif DETECT == "webcam":
        result_image = detect_objects_webcam(cap)
    else:
        print("Invalid argument provided. Use 'screen' or 'webcam'.")
        break

    # Display the result or save to disk
    if result_image is not None:
        try:
            cv2.imshow('YOLOv9 Object Detection', result_image)
            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except cv2.error as e:
            cv2.imwrite('output.jpg', result_image)
            print("cv2.imshow failed, saved the result as 'output.jpg'")
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()
sct.close()
