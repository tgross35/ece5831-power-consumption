"""Runner for the whole project using cli flags"""
import functools
import json
import os
import sys

import click

from crystalball.collect import run_collection
from crystalball.load import load_data
from crystalball.plot import plot_run
from crystalball.preprocess import preprocess_data


@click.option("--load", default=None, help="Load data into a database")
@click.option(
    "--process", is_flag=True, default=False, help="Run preprocessing on the data"
)
@click.option(
    "--collect",
    default=None,
    is_flag=True,
    help="Collect the data into numpy data types",
)
@click.option(
    "--train",
    default=None,
    help="Run the training. 'new' is an option, or you can specify a name",
)
@click.option(
    "--build", is_flag=True, default=False, help="Skip fitting, only run build"
)
@click.option("--fit", is_flag=True, default=False, help="Fitting only")
@click.option(
    "--predict",
    is_flag=True,
    default=False,
    help="Skip fitting, only run predictions",
)
@click.option(
    "--analyze",
    is_flag=True,
    default=False,
    help="Run analyzing on models",
)
@click.option(
    "--train-use-data",
    default=None,
    help="Specify dataset to use, othewise the latest discovered will be used",
)
@click.option(
    "--plot-results",
    is_flag=True,
    default=False,
    help="Run plot methods on training history",
)
@click.option(
    "--runname",
    default=None,
    help="Set run number",
)
@click.command()
def runtask(
    load,
    process,
    train,
    collect,
    train_use_data,
    build,
    fit,
    predict,
    analyze,
    plot_results,
    runname,
):
    """Main runner that handles flags"""
    print("Started a task")

    os.environ["CRYSTAL_MODEL_NAME"] = runname

    if load:
        load_data(load)

    if process:
        preprocess_data()

    if collect:
        run_collection()

    if train:
        # Lazy load tensorflow because it is sloooow
        from crystalball.train import run_training

        run_training(
            dataset=train_use_data,
            modelname=train,
            run_build=build,
            run_fit=fit,
            run_predict=predict,
            run_analyze=analyze,
        )

    if plot_results:
        print("Plotting results")
        plot_run()


if __name__ == "__main__":
    runtask()
