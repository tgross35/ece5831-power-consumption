"""Module to train our neural network"""

import inspect
import json
import math
import os
import types
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path

import numpy as np
import shap
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from tensorflow import keras
from termcolor import colored

from crystalball.collect import SET_DIRECTORY
from crystalball.constants import RUN_CONSTANTS
from crystalball.models import CombinedData, InputRegionInfo, session
from crystalball.namesgenerator import get_random_name

MODELS_DIRECTORY = None
EXPORT_DIRECTORY = None
LOG_MODELS_DIRECTORY = None
EPOCH_MULTIPLIER = None
BATCH_TYPES = None
NUM_EPOCHS = None
NUM_LAYERS = None


def config_constants():
    runname = os.environ["CRYSTAL_MODEL_NAME"]
    constants = RUN_CONSTANTS[runname]
    global MODELS_DIRECTORY, BATCH_TYPES, NUM_EPOCHS, NUM_LAYERS, EXPORT_DIRECTORY, EPOCH_MULTIPLIER, LOG_MODELS_DIRECTORY
    MODELS_DIRECTORY = constants["MODELS_DIRECTORY"]
    EXPORT_DIRECTORY = constants["EXPORT_DIRECTORY"]
    LOG_MODELS_DIRECTORY = constants["LOG_MODELS_DIRECTORY"]
    BATCH_TYPES = constants["BATCH_TYPES"]
    EPOCH_MULTIPLIER = constants["EPOCH_MULTIPLIER"]
    NUM_EPOCHS = constants["NUM_EPOCHS"]
    NUM_LAYERS = constants["NUM_LAYERS"]


# Will be import time, but close enough for us
START_DATE = datetime.now().isoformat()


def nickname_save(self):
    """Member function for models"""
    if not hasattr(self, "nickname"):
        self.nickname = get_random_name()

    self.save(self.savedir)
    plot_model(
        self,
        to_file=f"{self.savedir}/{self.nickname}.png",
        show_shapes=True,
        show_layer_names=True,
    )

    with open(f"{MODELS_DIRECTORY}/manifest.txt", "a") as f:
        f.write(f"{START_DATE} Generated model {self.nickname}\n")

    self.writesummary(f"Generated {START_DATE}. Summary:\n")
    self.writesummary(str(self.summary()))

    print(f"Saved model {self.nickname}")


def writesummary(self, summary):
    """Member functino to append to summary file"""
    print("Writing summary...")

    with open(f"{self.savedir}/{self.nickname}-summary.txt", "a") as f:
        f.write("\n\n")
        f.write(summary)


def writeinfo(self, **kw):
    """Member function to append to a json file."""
    path = Path(f"{self.savedir}/{self.nickname}-info.json")

    if not path.is_file():
        with open(path, "w") as f:
            json.dump({}, f, indent=4)

    with open(path, "r+") as f:
        file_data = json.load(f)
        for k, v in kw.items():
            file_data[k] = v
        f.seek(0)
        json.dump(file_data, f, indent=4)


def load_info(self) -> dict:
    """Member function to load info from a json file."""
    with open(f"{self.savedir}/{self.nickname}-info.json", "r") as f:
        data = json.load(f)
    return data


def update_model(model, name=None):
    """Add a name and helper method"""
    # Give our model a fun little name to remember it by
    model.nickname = get_random_name() if name is None else name
    model.savedir = f"{MODELS_DIRECTORY}/{model.nickname}"
    model.writesummary = types.MethodType(writesummary, model)
    model.writeinfo = types.MethodType(writeinfo, model)
    model.loadinfo = types.MethodType(load_info, model)
    model.nickname_save = types.MethodType(nickname_save, model)


def create_model(
    num_inputs: int,
    num_outputs: int,
    num_layers: int = 3,
    layer_space: str = "linear",
    name=None,
):
    """Create a tensorflow model

    num_layers INCLUDES one output layer"""

    print(
        f"Creating model with {num_inputs} inputs, {num_outputs} outputs, {num_layers} layers, and {layer_space} spacing"
    )

    # Need to flip so if 1 layer is specified we get the output size
    if layer_space == "log":
        lsizes = np.flip(np.geomspace(num_outputs, num_inputs, num_layers, dtype=int))
    else:
        # Assume linear
        lsizes = np.flip(np.linspace(num_outputs, num_inputs, num_layers, dtype=int))

    # Start out with a normalization layer
    # alternatively: keras.layers.Input(shape=(num_inputs,))
    # Norm gives us slightly better standard deviations
    layers_arr = [
        keras.layers.BatchNormalization(
            input_shape=(num_inputs,), name="batch_normalization"
        )
    ]

    # Add a layer for each given item
    for i, lsize in enumerate(lsizes):
        # If it's the last layer, no activation needed
        if i == num_layers:
            layers_arr.append(keras.layers.Dense(units=lsize, name="output"))
        else:
            layers_arr.append(
                keras.layers.Dense(units=lsize, activation="relu", name=f"dense_{i}")
            )

    model = keras.Sequential(layers_arr)

    model.compile(
        optimizer="adam",
        loss="MAPE",
        metrics=[
            tf.metrics.MeanAbsolutePercentageError(),
            tf.metrics.MeanSquaredError(),
            tf.metrics.MeanAbsoluteError(),
            tf.metrics.MeanSquaredLogarithmicError(),
            tf.metrics.RootMeanSquaredError(),
        ],
    )

    update_model(model, name)
    model.nickname_save()

    print(colored(f"Created model {model.nickname}", "green", attrs=["bold"]))
    print(model.summary())

    model.writesummary(
        inspect.cleandoc(
            f"""
            {datetime.now().isoformat()}
            Created model with parameters:

            - num_inputs: {num_inputs}
            - num_outputs: {num_outputs}
            - num_layers: {num_layers}
            - layer_space: {layer_space}
            """
        )
    )

    model.writeinfo(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        num_layers=num_layers,
        layer_space=layer_space,
    )

    return model


def load_model(name: str):
    """Load a model by its name"""
    filename = f"{MODELS_DIRECTORY}/{name}"
    model = keras.models.load_model(filename)
    update_model(model, name)

    print(f"Loaded model {model.nickname}")
    print(model.summary())

    return model


def fit_model(model, train_set_in, train_set_out, steps_per_epoch, epochs=10):
    """Task to start a model's fitting process."""
    print(colored(f"Started training {model.nickname}", "green"))
    log_dir = f"{LOG_MODELS_DIRECTORY}/{model.nickname}"
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )

    total_steps = steps_per_epoch * epochs
    in_len = train_set_in.shape[0]

    # Extend our arrays if needed
    if total_steps > in_len:
        in_width = train_set_in.shape[1]
        out_width = train_set_out.shape[1]
        train_set_in = np.repeat(train_set_in, math.ceil(total_steps / in_len)).reshape(
            (-1, in_width)
        )

        train_set_out = np.repeat(
            train_set_out, math.ceil(total_steps / in_len)
        ).reshape((-1, out_width))

    model.writesummary(
        inspect.cleandoc(
            f"""
            {datetime.now().isoformat()}
            Training model with parameters:
            - steps_per_epoch: {steps_per_epoch}
            - epochs: {epochs}
            """
        )
    )

    history = model.fit(
        train_set_in,
        train_set_out,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[tensorboard_callback],
    )

    model.nickname_save()
    np.save(f"{model.savedir}/history", history.history)

    model.writesummary(
        inspect.cleandoc(
            f"""
            {datetime.now().isoformat()}
            Training complete
            """
        )
    )

    print(colored(f"Finished training {model.nickname}", "green"))


def run_training(
    dataset: str = None,
    modelname: str = "new",
    run_fit=True,
    run_build=True,
    run_predict=True,
    run_analyze=True,
):
    """Create the modes and run them"""
    config_constants()

    if dataset is None:
        gl = glob(f"{SET_DIRECTORY}/*.n*pz")
        maxint = max([int(f.split("/")[-1].split(".")[0]) for f in gl])
        dataset = maxint

    with open(f"{SET_DIRECTORY}/{dataset}.shape", "r") as f:
        shapes = json.load(f)

    loadsets = np.load(f"{SET_DIRECTORY}/{dataset}.npz")

    datasets = {}
    for key, shape in shapes.items():
        datasets[key] = np.reshape(loadsets[key], shape)

    train_set_in = datasets["train_set_in"]
    train_set_out = datasets["train_set_out"]
    test_set_in = datasets["test_set_in"]
    test_set_out = datasets["test_set_out"]

    num_inputs = train_set_in.shape[1]
    input_len = train_set_in.shape[0]
    num_outputs = train_set_out.shape[1]
    default_epoch_size = math.floor(train_set_in.shape[0] / 10)

    print("Datasets loaded")

    models = []
    if modelname is not None and modelname not in ("new", "batch"):
        model = load_model(modelname)
        modelinfo = model.loadinfo()
        models = [
            {
                "model": model,
                "epochs": modelinfo["epochs"],
                "epoch_size": modelinfo["epoch_size"],
            }
        ]
    elif modelname is not None and modelname == "batch":
        if run_build:
            models = [
                {
                    "model": create_model(
                        num_inputs=num_inputs,
                        num_outputs=num_outputs,
                        num_layers=nl,
                        layer_space=ls,
                    ),
                    "epochs": ep,
                    "epoch_size": math.floor(em * input_len),
                }
                for nl in NUM_LAYERS
                for ls in BATCH_TYPES
                for ep in NUM_EPOCHS
                for em in EPOCH_MULTIPLIER
            ]

            for m in models:
                m["model"].writeinfo(
                    epoch_size=m["epoch_size"],
                    epochs=m["epochs"],
                )
        else:
            globs = glob(f"{MODELS_DIRECTORY}/*")
            modelnames = [g.split("/")[-1] for g in globs if "manifest.txt" not in g]
            models = []
            for modelname in modelnames:
                model = load_model(modelname)
                modelinfo = model.loadinfo()
                m = {
                    "model": model,
                    "epochs": modelinfo["epochs"],
                    "epoch_size": modelinfo["epoch_size"],
                }

                models.append(m)

        print(colored(f"Created {len(models)} models", "blue"))
    else:
        # Create single new model if modelname is new
        if run_build:
            models = [
                {"model": create_model(num_inputs=num_inputs, num_outputs=num_outputs)}
            ]

    if run_fit:
        # Collect all our models to run
        for m in models:
            model = m["model"]
            print(colored(f"Collecting model {model.nickname}", "green"))

            fit_model(
                model,
                train_set_in,
                train_set_out,
                steps_per_epoch=m.get("epoch_size", default_epoch_size),
                epochs=m.get("epochs", 10),
            )
    else:
        print(colored("Skipping fitting", "green"))

    if run_predict:
        for m in models:
            model = m["model"]
            print(model.summary())

            predictions = model.predict(test_set_in)
            pct = 100 * (predictions - test_set_out) / test_set_out
            pct_avg = np.average(pct, axis=0)
            pct_std = np.std(pct, axis=0)

            np.savez(
                f"{model.savedir}/outputs",
                predictions=predictions,
                pct=pct,
                pct_avg=pct_avg,
                pct_std=pct_std,
            )

            output = [
                f"Model {model.nickname} trained with averages:",
                str(pct_avg),
                "And standard deviations:",
                str(pct_std),
            ]

            for item in output:
                print(item)
                model.writesummary(item)

            model.writeinfo(pct_avg=pct_avg.tolist(), pct_std=pct_std.tolist())

    if run_analyze:
        train_set_in
        # shap doesn't work for the latest tf :(
        # background = train_set_in[
        #     np.random.choice(train_set_in.shape[0], 100, replace=False)
        # ]
        id = np.identity(test_set_in.shape[1])
        background = np.multiply(id, np.average(test_set_in, axis=0))

        for m in models:
            model = m["model"]

            pd = model.predict(background)
            summed = np.sum(pd, axis=1)
            # avg=np.average(summed)
            contrib = summed / np.sum(summed)

            # e = shap.DeepExplainer(model, background)
