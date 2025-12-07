"""
Simple test script to compare YOLOv12 baseline vs YOLOv12-CBAM.
This script trains both models and evaluates them on the same dataset.
"""
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Handle tomllib import (Python 3.11+) or fallback to toml
try:
    import tomllib
    USE_TOMLLIB = True
except ModuleNotFoundError:
    import toml as tomllib
    USE_TOMLLIB = False

# from src.models.yolo_wrapper import YoloWrapper  # Baseline already trained
from src.models.yolo_cbam_wrapper import YoloCbamWrapper


def read_configs():
    """Load configuration from config.toml"""
    if USE_TOMLLIB:
        # Python 3.11+ tomllib uses binary mode
        with open('config/config.toml', 'rb') as f:
            config_data = tomllib.load(f)
    else:
        # toml library uses text mode
        with open('config/config.toml', 'r') as f:
            config_data = tomllib.load(f)
    return config_data


def test_cbam_vs_baseline():
    """
    Compare YOLOv12-CBAM vs baseline YOLOv12 on source dataset.
    """
    # Load configs
    configs = read_configs()

    # Setup paths
    data_yaml = os.path.join(configs["source_root"], "data.yaml")
    experiment_root = os.path.join(os.getcwd(), 'experiments', 'cbam')
    os.makedirs(experiment_root, exist_ok=True)

    print("=" * 80)
    print("YOLOv12 CBAM vs Baseline Comparison")
    print("=" * 80)

    # ========== Train and evaluate baseline YOLOv12 ==========
    # print("\n[1/2] Training baseline YOLOv12...")
    # yolo_baseline = YoloWrapper(
    #     weigths=configs['yolo_weights'],
    #     img_size=configs['img_size']
    # )

    # yolo_baseline.train_model(
    #     data_yaml=data_yaml,
    #     batch_size=configs['batch_size'],
    #     epochs=configs['epochs'],
    #     project=experiment_root,
    #     name='yolo12_baseline'
    # )

    # print("\n[1/2] Evaluating baseline YOLOv12...")
    # baseline_metrics = yolo_baseline.evaluate(
    #     data_yaml=data_yaml,
    #     project=experiment_root,
    #     name='yolo12_baseline_eval'
    # )

    # ========== Train and evaluate YOLOv12-CBAM ==========
    print("\n[2/2] Training YOLOv12-CBAM...")
    yolo_cbam = YoloCbamWrapper(
        model_config=configs['yolo_cbam_config'],
        img_size=configs['img_size']
    )

    yolo_cbam.train_model(
        data_yaml=data_yaml,
        batch_size=configs['batch_size'],
        epochs=configs['epochs'],
        project=experiment_root,
        name='yolo12_cbam'
    )

    print("\n[2/2] Evaluating YOLOv12-CBAM...")
    cbam_metrics = yolo_cbam.evaluate(
        data_yaml=data_yaml,
        project=experiment_root,
        name='yolo12_cbam_eval'
    )

    # ========== Print CBAM results ==========
    print("\n" + "=" * 80)
    print("YOLOv12-CBAM RESULTS")
    print("=" * 80)
    cbam_map50 = cbam_metrics.box.map50
    cbam_map = cbam_metrics.box.map
    print(f"mAP@0.5:      {cbam_map50:.4f}")
    print(f"mAP@0.5:0.95: {cbam_map:.4f}")
    print("=" * 80)

    return {
        'cbam': cbam_metrics
    }

    # ========== Comparison (commented out - baseline already trained) ==========
    # print("\n" + "=" * 80)
    # print("RESULTS COMPARISON")
    # print("=" * 80)
    # print(f"{'Model':<25} {'mAP@0.5':<15} {'mAP@0.5:0.95':<15}")
    # print("-" * 80)
    # baseline_map50 = baseline_metrics.box.map50
    # baseline_map = baseline_metrics.box.map
    # cbam_map50 = cbam_metrics.box.map50
    # cbam_map = cbam_metrics.box.map
    # print(f"{'YOLOv12 Baseline':<25} {baseline_map50:<15.4f} "
    #       f"{baseline_map:<15.4f}")
    # print(f"{'YOLOv12-CBAM':<25} {cbam_map50:<15.4f} "
    #       f"{cbam_map:<15.4f}")
    # print("-" * 80)

    # # Calculate improvement
    # map50_improvement = (
    #     (cbam_map50 - baseline_map50) / baseline_map50 * 100
    # )
    # map_improvement = (cbam_map - baseline_map) / baseline_map * 100

    # print("\nCBAM Improvement:")
    # print(f"  mAP@0.5:      {map50_improvement:+.2f}%")
    # print(f"  mAP@0.5:0.95: {map_improvement:+.2f}%")
    # print("=" * 80)

    # return {
    #     'baseline': baseline_metrics,
    #     'cbam': cbam_metrics
    # }


if __name__ == "__main__":
    results = test_cbam_vs_baseline()
