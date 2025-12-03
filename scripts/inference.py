import os
import cv2
import numpy as np
import tomllib
import matplotlib.pyplot as plt
from src.models.rtdeter_wrapper import RTdeterWrapper
from src.models.yolo_wrapper import YoloWrapper

'''
Iterates through the test set directory of all target domains. Run inference on each test set image with both models,
plots bounding boxes, and saves images.

experiment.py (python -m scripts.experiment from root dir) should be run first to generate target domains and train the models.
'''


def inference(img_path, output_dir, yolo_model, rtdeter_model):
    '''
    Performs inference on a single image using both yolo_model and rtdeter_model. Plots bounding boxes for both predictions onto image.
    Saves the image to output_dir.
    
    :param img_path: path to image to perform inference on
    :param output_dir: path to save image after bounding boxes have been applied
    :param yolo_model: YOLOv12 object with trained weights to use for inference
    :param rtdeter_model: RTDETER object with trained weights to use for inference
    '''
    # TODO Make sure to run python -m scripts.experiment from root directory first to generate target datasets and train models
    '''
    1. Load the image (e.g., with cv2.imread(img_path)).
    2. Run both models:
         yolo_results   = yolo_model.inference(img_path)
         rtdetr_results = rtdeter_model.inference(img_path)
       Each returns a list; use results[0].
    3. From results[0], use:
         .boxes.xyxy  -> [x1, y1, x2, y2]
         .boxes.conf  -> confidence
         .boxes.cls   -> class id (0='character', 1='number-plates')
    4. Draw YOLO boxes (e.g., red) and RT-DETR boxes (e.g., blue) on the image.
    5. Save the image to output_dir with the same filename as img_path.

    '''
    return None




def main():
    # read .toml
    with open('config/config.toml', 'rb') as f:
        configs = tomllib.load(f)
    
    source_data_root = configs["source_root"]
    # verify source data exists
    source_check = os.path.join(source_data_root, "data.yaml")
    if not os.path.exists(source_check):
        print("data.yaml from source dataset not found. See README for data set download steps.")
        return
    
    # verify models have been trained
    experiments_path = os.path.join(os.getcwd(),"experiments")
    if not os.path.exists(experiments_path) or len(os.listdir(experiments_path))==0:
        print("No trained models. Run python -m scripts.experiment to train models.")
        return
    
    # get trained weights
    yolo_weights = ""
    rtdeter_weights = ""
    for dir in os.listdir(experiments_path):
        yolo_path = os.path.join(experiments_path, dir, "YOLO_train", "weights", "best.pt")
        rtdeter_path = os.path.join(experiments_path, dir, "RTDETER_train", "weights", "best.pt")
        if os.path.exists(yolo_path) and os.path.exists(rtdeter_path):
            yolo_weights = yolo_path
            rtdeter_weights = rtdeter_path
            break
    if yolo_weights=="" or rtdeter_weights=="":
        print("No weights found. Run python -m scripts.experiment to train models")
        return
    
    # instantiate models
    yolo_model = YoloWrapper(yolo_weights)
    rtdeter_model = RTdeterWrapper(rtdeter_weights)

    # Create output directory
    output_dir_root = os.path.join(os.getcwd(),"inference_images")
    os.makedirs(output_dir_root,exist_ok=True)


    # iterate through target_data directories test directories and performe inference
    target_data_root = configs["target_root"]
    for data_dir in os.listdir(target_data_root):
        output_dir = os.path.join(output_dir_root,data_dir)
        os.makedirs(output_dir,exist_ok=True)
        test_dir_path = os.path.join(target_data_root,data_dir,"test","images")
        if not os.path.exists(test_dir_path):
            continue
        for img in os.listdir(test_dir_path):
            if not img.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(test_dir_path,img)
            if os.path.exists(img_path):
                inference(img_path, output_dir, yolo_model, rtdeter_model)
            else:
                continue



if __name__ == "__main__":
    main()