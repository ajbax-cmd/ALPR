import os
import tomllib
import csv
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data.degradation import Degradation
from src.models.rtdeter_wrapper import RTdeterWrapper
from src.models.yolo_wrapper import YoloWrapper


def main():
    base_line_results_path = os.path.join(os.getcwd(), "base_line_results")
    os.makedirs(base_line_results_path,exist_ok=True)

    train_yaml = "src/data/source_data/data.yaml"
    eval_yaml = "src/data/source_data/data.yaml"
    # train and eval yolo model source->source
    yolo_model = YoloWrapper()
    yolo_model.train_model(train_yaml, 4, 20,project=base_line_results_path,name='YOLO_train')
    yolo_metrics = yolo_model.evaluate(eval_yaml,project=base_line_results_path,name='YOLO_eval')

    rtdeter_model = RTdeterWrapper()
    rtdeter_model.train_model(train_yaml, 4, 20,project=base_line_results_path,name='RTDETER_train')
    rtdeter_metrics = rtdeter_model.evaluate(eval_yaml,project=base_line_results_path,name='RTDETER_eval')

    print(f"YOLO mAP50: {yolo_metrics.box.map50}")
    print(f"YOLO mAP50:95: {yolo_metrics.box.map}")
    print(f"RTDETER mAP50: {rtdeter_metrics.box.map50}")
    print(f"RTDETER mAP50:95: {rtdeter_metrics.box.map}")


if __name__ == "__main__":
    main()