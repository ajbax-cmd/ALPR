## Installation
To set up the conda env, navigate to the root directory of the project (ALPR) and run:
```
conda env create -f env.yml
conda activate alpr
```

## Dataset - UK Number Plate Recognision
The dataset can be downloaded from:

https://universe.roboflow.com/recognision-datasets/uk-number-plate-recognision/dataset/2

1. Select YOLOv11 download format and download as a zip file.
2. Unpack the dataset under the ALPR/src/data directory.
3. Rename the root direcotry of the dataset "source_data" so that the directory structure is ALPR/src/data/source_data

## Running an Experiment
1. Set up the ALRP/config/config.toml file with the desired valid parameters and save.
2. From the root directory:
```
python -m scripts.experiment
```
3. Cross Domain eval results (source->target and target->source) are logged under ALPR/experiments directory.
4. Line graphs and Heat Maps are generated and saved under ALPR/plots

## Inference
1. Run an experiment first to generate target domain datasets and train models first
2. From the root directory:
```
python -m scripts.inference
```
3. Test images with bounding box coordinates of both models plotted are saved at ALPR/inference_images

