from ultralytics import YOLO
from pathlib import Path
import torch

# Import and register CBAM module
from .cbam import CBAM

# Register CBAM with Ultralytics in multiple places to ensure it's found
import ultralytics.nn.modules as nn_modules

# Add to nn.modules namespace
nn_modules.CBAM = CBAM

# Add to the module's __dict__ to make it importable
if 'CBAM' not in nn_modules.__dict__:
    nn_modules.__dict__['CBAM'] = CBAM

# Also register in ultralytics.nn.tasks globals where parse_model looks
import ultralytics.nn.tasks as tasks
tasks.CBAM = CBAM


class YoloCbamWrapper():
    """
    Wrapper for YOLOv12 with CBAM (Convolutional Block Attention Module).
    Uses custom config file with CBAM modules integrated into the backbone.
    """
    def __init__(self, model_config="config/yolo12-cbam.yaml", weights=None, img_size=640, scale='l'):
        """
        Initialize YOLOv12-CBAM model.

        Args:
            model_config (str): Path to CBAM-enhanced YOLO config file
            weights (str, optional): Path to pretrained weights. If None, trains from scratch.
            img_size (int): Input image size
            scale (str): Model scale ('n', 's', 'm', 'l', 'x'). Default: 'l' for large
        """
        self.model_config = model_config
        self.img_size = img_size
        self.scale = scale
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Read YAML and inject scale
        import yaml
        with open(model_config, 'r') as f:
            cfg = yaml.safe_load(f)

        # Override scale in config
        cfg['scale'] = scale

        # Create model with modified config
        print(f"Initializing YOLOv12-CBAM with scale='{scale}' from config: {model_config}")

        # Write temporary config file with scale
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(cfg, f)
            temp_config = f.name

        try:
            # Load model from temporary config
            self.model = YOLO(temp_config)
            print(f"Model initialized with {sum(p.numel() for p in self.model.model.parameters())/1e6:.1f}M parameters")
        finally:
            # Clean up temp file
            if os.path.exists(temp_config):
                os.unlink(temp_config)

        self.model.to(self.device)

    def train_model(self, data_yaml, batch_size, epochs, project, name):
        """
        Train the CBAM-enhanced YOLO model.

        Args:
            data_yaml (str): Path to dataset YAML file
            batch_size (int): Batch size for training
            epochs (int): Number of training epochs
            project (str): Project directory
            name (str): Experiment name

        Returns:
            Training results object
        """
        train_results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=self.img_size,
            batch=batch_size,
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            scale=0.0,
            device=0 if torch.cuda.is_available() else 'cpu',  # Use GPU 0 if available
            project=project,
            name=name
        )
        return train_results

    def evaluate(self, data_yaml, project, name):
        """
        Evaluate the CBAM-enhanced YOLO model.

        Args:
            data_yaml (str): Path to dataset YAML file
            project (str): Project directory
            name (str): Experiment name

        Returns:
            Validation metrics object
        """
        metrics = self.model.val(
            data=data_yaml,
            imgsz=self.img_size,
            device=0 if torch.cuda.is_available() else 'cpu',
            project=project,
            name=name
        )
        return metrics

    def inference(self, x, show=False, save=False):
        """
        Run inference on input images.

        Args:
            x: Input image(s)
            show (bool): Whether to display results
            save (bool): Whether to save results

        Returns:
            Detection results
        """
        results = self.model(
            x,
            imgsz=self.img_size,
            device=0 if torch.cuda.is_available() else 'cpu',
            show=show,
            save=save,
        )
        return results
