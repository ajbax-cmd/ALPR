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


def read_configs():
    with open('config/config.toml', 'rb') as f:
        config_data = tomllib.load(f)
    return config_data

def append_row(root_dir, row):
    global_csv_path = os.path.join(root_dir, 'global_results.csv')
    global_exists = os.path.exists(global_csv_path)

    field_names = [
        "exp_name",
        "model",
        "train_on",
        "eval_on",
        "kernel",
        "scale",
        "sigma",
        "map50",
        "map5095",
        "f1",
    ]
    with open(global_csv_path,'a',newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        if not global_exists:
             writer.writeheader()
        writer.writerow(row)

def generate_plots(csv_path, configs):
    plots_path = os.path.join(os.getcwd(),"plots")
    os.makedirs(plots_path, exist_ok=True)
    df = pd.read_csv(csv_path)
    median_scale = configs["scale"][len(configs["scale"])//2]
    median_sigma = configs["sigma"][len(configs["sigma"])//2]
    yolo_ret_den = configs["map95_YOLO"]
    rtdeter_ret_den = configs["map95_RTDETER"]

    k0 = configs["kernel"][0]
    # plot mAP vs sigma with fixed scale
    plot_sigma_fixed_scale(df, kernel=k0, scale=median_scale,
                    out_path=f"plots/map95_vs_sigma_{k0}_scale{median_scale}.png")
    
    # plot mAP vs scale with fixed sigma
    plot_scale_fixed_sigma(df, kernel=k0, sigma=median_sigma,
                           out_path=f"plots/map95_vs_scale_{k0}_sigma{median_sigma}.png")

    # plot heatmaps RTdeter mAP0.95 - YOLOv12 mAP0.95 for each scale sigma pair
    plot_heat_map(df,kernel=k0, train_on="source", eval_on="target", out_path=f"plots/heatmap_{k0}_source_target.png")
    plot_heat_map(df,kernel=k0, train_on="target", eval_on="source", out_path=f"plots/heatmap_{k0}_target_source.png")
    
    # plot retention
    plot_percent_drop_gridline(df, configs, kernel=k0, out_path=f"plots/retention_{k0}.png")

    if len(configs["kernel"]) > 1:
        k1 = configs["kernel"][1]
        # plot mAP vs sigma with fixed scale
        plot_sigma_fixed_scale(df, kernel=k1, scale=median_scale,
                out_path=f"plots/map95_vs_sigma_{k1}_scale{median_scale}.png")
        # plot mAP vs scale with fixed sigma
        plot_scale_fixed_sigma(df, kernel=k1, sigma=median_sigma,
                out_path=f"plots/map95_vs_scale_{k1}_sigma{median_sigma}.png")
        # plot heatmaps RTdeter mAP0.95 - YOLOv12 mAP0.95 for each scale sigma pair
        plot_heat_map(df,kernel=k1, train_on="source", eval_on="target", out_path=f"plots/heatmap_{k1}_source_target.png")
        plot_heat_map(df,kernel=k1, train_on="target", eval_on="source", out_path=f"plots/heatmap_{k1}_target_source.png")
        # plot retention
        plot_percent_drop_gridline(df, configs, kernel=k1, out_path=f"plots/retention_{k1}.png")


def plot_sigma_fixed_scale(df, kernel, scale, out_path=None):
    sub_df = df[(df["kernel"] == kernel) & (df["scale"] == scale)]
    if sub_df.empty:
        return
    groups = sub_df.groupby(["model", "train_on", "eval_on"])
    plt.figure()

    for (model, train_on, eval_on), g in groups:
        g = g.sort_values("sigma")

        label = f"{model} {train_on}->{eval_on}"
        plt.plot(g["sigma"], g["map5095"], marker="o", label=label)

    plt.xlabel("sigma (gaussian noise std)")
    plt.ylabel("mAP@0.5:0.95")
    plt.title(f"mAP@0.5:0.95 vs sigma\nkernel={kernel}, scale={scale}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if out_path is not None:
        plt.savefig(out_path, bbox_inches="tight")
    else:
        plt.show()

def plot_scale_fixed_sigma(df, kernel, sigma, out_path=None):
    sub_df = df[(df["kernel"] == kernel) & (df["sigma"] == sigma)]
    if sub_df.empty:
        return
    groups = sub_df.groupby(["model", "train_on", "eval_on"])
    plt.figure()

    for(model, train_on, eval_on), g in groups:
        g = g.sort_values("scale")

        label = f"{model}{train_on}->{eval_on}"
        plt.plot(g["scale"], g["map5095"], marker="o", label=label)

    plt.xlabel("scale (downsampling factor)")
    plt.ylabel("mAP@0.5:0.95")
    plt.title(f"mAP@0.5:0.95 vs scale\nkernel={kernel}, sigma={sigma}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if out_path is not None:
        plt.savefig(out_path, bbox_inches="tight")
    else:
        plt.show()

def plot_heat_map(df, kernel, train_on="source",eval_on="target",out_path=None):
    sub = df[(df["kernel"] == kernel)&(df["train_on"] == train_on)&(df["eval_on"] == eval_on)]
    if sub.empty:
        return
    
    table = sub.pivot_table(index=["sigma", "scale"],columns="model",values="map5095")
    required = ["yolov12l", "rtdeter-l"]
    missing = [m for m in required if m not in table.columns]
    if missing:
        return
    
    table = table.dropna(subset=required)
    if table.empty:
        return
    # compute delta between RTDETERmAP95 - YOLOmAP95 for each scale, sigma pair
    table["delta"] = table["rtdeter-l"] - table["yolov12l"]
    delta_mat = table["delta"].unstack("scale")
    delta_mat = delta_mat.sort_index(axis=0)  # sort by sigma
    delta_mat = delta_mat.sort_index(axis=1)  # sort by scale

    sigmas = delta_mat.index.values
    scales = delta_mat.columns.values
    Z = delta_mat.values

    vmax = np.nanmax(np.abs(Z))
    if vmax == 0 or np.isnan(vmax):
        vmax = 1.0  

    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        Z,
        origin="lower",
        aspect="auto",
        cmap="RdBu",
        vmin=-vmax,
        vmax=vmax,
        extent=[scales.min(), scales.max(), sigmas.min(), sigmas.max()],
    )

    plt.colorbar(im, label="delta mAP@0.5:0.95 (RT-DETR - YOLO)")
    plt.xlabel("scale (downsampling factor)")
    plt.ylabel("sigma (gaussian noise std)")
    plt.title(f"RT-DETR advantage over YOLO\nkernel={kernel}, {train_on}->{eval_on}")

    plt.xticks(scales)
    plt.yticks(sigmas)

    if out_path is not None:
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_percent_drop_gridline(df, configs, kernel, out_path=None):
    yolo_baseline = configs["map95_YOLO"]
    rtdeter_baseline = configs["map95_RTDETER"]

    baseline_map95 = {
        "yolov12l": yolo_baseline,
        "rtdeter-l": rtdeter_baseline
    }

    sub = df[
        (df["kernel"] == kernel) &
        (df["train_on"] == "source") &
        (df["eval_on"] == "target")
    ].copy()

    if sub.empty:
        return

    sub = sub[sub["model"].isin(["yolov12l", "rtdeter-l"])]


    sub["baseline"] = sub["model"].map(baseline_map95)

    sub = sub.dropna(subset=["baseline"])
    if sub.empty:
        return

    sub["retention"] = sub["map5095"] / sub["baseline"]
    sub["pct_drop"] = 100 * (1 - sub["retention"])

    sub["x_label"] = sub.apply(lambda r: f"s={r['scale']}, σ={int(r['sigma'])}", axis=1)

    sub = sub.sort_values(["scale", "sigma"])
    x_order = sub["x_label"].drop_duplicates().tolist()

    plt.figure(figsize=(8, 3.5))
    for model, g in sub.groupby("model"):
        g = g.set_index("x_label").loc[x_order].reset_index()
        label = "YOLOv12" if model == "yolov12l" else "RT-DETR"
        plt.plot(g["x_label"], g["pct_drop"], marker="o", label=label)

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("% drop from source baseline (mAP@0.5:0.95)")
    plt.title(f"Source→Target percent drop across (scale, σ)\nkernel={kernel}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def run_experiment():
    # read .toml
    configs = read_configs()

    # create experiment root directory
    experiment_root_path = os.path.join(os.getcwd(),'experiments')
    os.makedirs(experiment_root_path,exist_ok=True)

    for kernel in configs['kernel']:
        for scale in configs['scale']:
            for sigma in configs['sigma']:
                # create experiment directory
                name = f"{kernel}_scale_{scale}_sigma_{sigma}"
                experiment_path = os.path.join(experiment_root_path, name)
                os.makedirs(experiment_path, exist_ok=True)

                # generate target domain
                data_out=Degradation.make_new_domain(configs['source_root'], configs['target_root'], sigma, scale, kernel)

                # source->target------------------------------------------
                train_yaml = os.path.join(configs["source_root"], "data.yaml")
                eval_yaml = os.path.join(data_out, "data.yaml")
                # train and eval yolo model source->target
                yolo_model = YoloWrapper(configs['yolo_weights'], configs['img_size'])
                yolo_model.train_model(train_yaml, configs['batch_size'], configs['epochs'],project=experiment_path,name='YOLO_train')
                yolo_metrics = yolo_model.evaluate(eval_yaml,project=experiment_path,name='YOLO_eval')
                
                yolo_row = {
                "exp_name": name,
                "model": "yolov12l",
                "train_on": "source",
                "eval_on": "target",
                "kernel": kernel,
                "scale": scale,
                "sigma": sigma,
                "map50": yolo_metrics.box.map50,
                "map5095": yolo_metrics.box.map,
                "f1": getattr(yolo_metrics, "f1", None),  
                }

                append_row(experiment_root_path, yolo_row )

                # train and eval RTdeter model source->target
                rtdeter_model = RTdeterWrapper(configs['rtdeter_weights'], configs['img_size'])
                rtdeter_model.train_model(train_yaml, configs['batch_size'], configs['epochs'],project=experiment_path,name='RTDETER_train')
                rtdeter_metrics = rtdeter_model.evaluate(eval_yaml,project=experiment_path,name='RTDETER_eval')

                rtdeter_row = {
                "exp_name": name,
                "model": "rtdeter-l",
                "train_on": "source",
                "eval_on": "target",
                "kernel": kernel,
                "scale": scale,
                "sigma": sigma,
                "map50": rtdeter_metrics.box.map50,
                "map5095": rtdeter_metrics.box.map,
                "f1": getattr(rtdeter_metrics, "f1", None),  
                }

                append_row(experiment_root_path, rtdeter_row )

                # target->source------------------------------------------
                train_yaml = os.path.join(data_out, "data.yaml")
                eval_yaml = os.path.join(configs["source_root"], "data.yaml")

                # train and eval yolo model target->source
                yolo_model = YoloWrapper(configs['yolo_weights'], configs['img_size'])
                yolo_model.train_model(train_yaml, configs['batch_size'], configs['epochs'],project=experiment_path,name='YOLO_train')
                yolo_metrics = yolo_model.evaluate(eval_yaml,project=experiment_path,name='YOLO_eval')
                
                yolo_row = {
                "exp_name": name,
                "model": "yolov12l",
                "train_on": "target",
                "eval_on": "source",
                "kernel": kernel,
                "scale": scale,
                "sigma": sigma,
                "map50": yolo_metrics.box.map50,
                "map5095": yolo_metrics.box.map,
                "f1": getattr(yolo_metrics, "f1", None),  
                }


                append_row(experiment_root_path, yolo_row )

                # train and eval RTdeter model target->source
                rtdeter_model = RTdeterWrapper(configs['rtdeter_weights'], configs['img_size'])
                rtdeter_model.train_model(train_yaml, configs['batch_size'], configs['epochs'],project=experiment_path,name='RTDETER_train')
                rtdeter_metrics = rtdeter_model.evaluate(eval_yaml,project=experiment_path,name='RTDETER_eval')

                rtdeter_row = {
                "exp_name": name,
                "model": "rtdeter-l",
                "train_on": "target",
                "eval_on": "source",
                "kernel": kernel,
                "scale": scale,
                "sigma": sigma,
                "map50": rtdeter_metrics.box.map50,
                "map5095": rtdeter_metrics.box.map,
                "f1": getattr(rtdeter_metrics, "f1", None),  
                }

                append_row(experiment_root_path, rtdeter_row )

                # log configs used
                config_out_path = os.path.join(experiment_path, "configs_used.json")
                with open(config_out_path, "w") as f:
                    json.dump(configs, f, indent=2)

    return experiment_root_path, configs

def main():
    # read .toml
    #configs = read_configs()
    # create experiment root directory
    #csv_path = os.path.join(os.getcwd(),'experiments')


    csv_path, configs = run_experiment()
    csv_path = os.path.join(csv_path,'global_results.csv')
    generate_plots(csv_path, configs)

if __name__ == "__main__":
    main()
