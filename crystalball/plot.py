import json
import os
from glob import glob
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

from crystalball.constants import RUN_CONSTANTS

STYLES = ["-", "--", "-.", ":"]
MODELS_DIRECTORY = None
EXPORT_DIRECTORY = None
BATCH_TYPES = None
NUM_EPOCHS = None
NUM_LAYERS = None


def config_constants():
    runname = os.environ["CRYSTAL_MODEL_NAME"]
    constants = RUN_CONSTANTS[runname]
    global MODELS_DIRECTORY, BATCH_TYPES, NUM_EPOCHS, NUM_LAYERS, EXPORT_DIRECTORY
    MODELS_DIRECTORY = constants["MODELS_DIRECTORY"]
    EXPORT_DIRECTORY = constants["EXPORT_DIRECTORY"]
    BATCH_TYPES = constants["BATCH_TYPES"]
    NUM_EPOCHS = constants["NUM_EPOCHS"]
    NUM_LAYERS = constants["NUM_LAYERS"]


def get_models():
    globs = glob(f"{MODELS_DIRECTORY}/*")
    modeldirs = [g for g in globs if "manifest.txt" not in g]
    # rem = "manifest.txt"
    # modelnames.remove(rem) if rem in modelnames else None
    # plot_arrs = np.zeros()

    models = []

    for modeldir in modeldirs:
        modelname = modeldir.split("/")[-1]
        print(f"Loading {modelname}")
        history = np.load(f"{modeldir}/history.npy", allow_pickle=True).item()
        pct_data = np.load(f"{modeldir}/outputs.npz")
        with open(f"{modeldir}/{modelname}-info.json", "r") as f:
            modelinfo = json.load(f)

        for k, v in history.items():
            modelinfo[k] = v

        for k, v in pct_data.items():
            modelinfo[k] = v

        modelinfo["nickname"] = modelname
        modelinfo["titlename"] = modelname.replace("_", " ").title()
        modelinfo["modeldir"] = modeldir

        models.append(modelinfo)
    return models


def plot_history(models):
    """Plot history of a list of models"""
    print("Plotting History")

    def semilogconfig(_plt):
        _plt.figure(dpi=1200)
        _plt.grid(True)
        _plt.xlabel("Epoch")
        _plt.ylabel("Percent Error")

    def unique_legend(_plt):
        handles, labels = _plt.gca().get_legend_handles_labels()
        # labels will be the keys of the dict, handles will be values
        temp = {k: v for k, v in zip(labels, handles)}
        try:
            # try sort by numbers first
            temp = {k: temp[k] for k in sorted(temp, key=lambda x: float(x))}
        except ValueError:
            temp = {k: temp[k] for k in sorted(temp)}

        _plt.legend(temp.values(), temp.keys(), loc="best")

    ### Plot BATCH_TYPES differences
    semilogconfig(plt)
    for i, type in enumerate(BATCH_TYPES):
        data = [
            m["mean_absolute_percentage_error"]
            for m in models
            if m["layer_space"] == type
        ]

        x = np.array([i for i in range(0, len(data[0]))])
        xaxis = np.linspace(x.min(), x.max(), 30)
        for j, d in enumerate(data):
            smooth = make_interp_spline(x, d)
            plt.xticks(x)
            plt.semilogy(
                xaxis,
                smooth(xaxis),
                color="#8EB6F8",
                linestyle=STYLES[i],
                alpha=0.85,
                label=f"{type}",
            )

    plt.title("Batch Types")
    unique_legend(plt)
    plt.savefig(f"{EXPORT_DIRECTORY}/batch_types.png")

    ### Plot EPOCH_MULTIPLIER differences
    semilogconfig(plt)
    epoch_sizes = {m["epoch_size"] for m in models}
    msort = sorted(models, key=lambda x: x["epoch_size"])
    for i, size in enumerate(epoch_sizes):
        data = [
            m["mean_absolute_percentage_error"]
            for m in msort
            if m["epoch_size"] == size
        ]

        x = np.array([i for i in range(0, len(data[0]))])
        xaxis = np.linspace(x.min(), x.max(), 30)
        for j, d in enumerate(data):
            smooth = make_interp_spline(x, d)
            plt.xticks(x)
            plt.semilogy(
                xaxis,
                smooth(xaxis),
                color="#8EB6F8",
                linestyle=STYLES[i],
                alpha=0.85,
                label=f"{size}",
            )

    plt.title("Epoch Size")
    unique_legend(plt)
    plt.savefig(f"{EXPORT_DIRECTORY}/epoch_size.png")

    ### Plot NUM_LAYERS differences
    semilogconfig(plt)
    for i, count in enumerate(NUM_LAYERS):
        data = [
            m["mean_absolute_percentage_error"]
            for m in models
            if m["num_layers"] == count
        ]

        x = np.array([i for i in range(0, len(data[0]))])
        xaxis = np.linspace(x.min(), x.max(), 30)
        for j, d in enumerate(data):
            smooth = make_interp_spline(x, d)
            plt.xticks(x)
            plt.semilogy(
                xaxis,
                smooth(xaxis),
                color="#8EB6F8",
                linestyle=STYLES[i],
                alpha=0.85,
                label=f"{count-1}",
            )

    plt.title("Hidden layer Count")
    unique_legend(plt)
    plt.savefig(f"{EXPORT_DIRECTORY}/layer_count.png")

    ### Plot with layer names
    semilogconfig(plt)
    xsize = len(models[0]["mean_absolute_percentage_error"])
    x = np.array([i for i in range(0, xsize)])
    xaxis = np.linspace(x.min(), x.max(), 30)
    for m in models:
        d = m["mean_absolute_percentage_error"]
        smooth = make_interp_spline(x, d)
        plt.xticks(x)
        plt.semilogy(
            xaxis,
            smooth(xaxis),
            alpha=0.9,
            label=f"{m['titlename']} ({m['num_layers']-1})",
        )

    plt.title("Training Algorithms Used and Hidden Layer Count")
    unique_legend(plt)
    plt.savefig(f"{EXPORT_DIRECTORY}/labeled.png")


def plot_error(models):
    print("Plotting errors")
    hours = np.array([i for i in range(1, 49)])

    for model in models:
        plt.figure(dpi=1200)
        plt.grid(True)
        plt.xlabel("Hours in future")
        plt.ylabel("Average Percent Error")
        plt.title(f"Percent Error by Hour ({model['titlename']})")

        pct_avg = model["pct_avg"]
        pct_std = model["pct_std"]
        plt.plot(hours, pct_avg)
        plt.errorbar(hours, pct_avg, pct_std)
        plt.savefig(f"{model['modeldir']}/{model['nickname']}-error.png")


def plot_run():
    config_constants()
    plt.rcParams["font.family"] = "Times New Roman"
    models = get_models()
    plot_history(models)
    plot_error(models)
