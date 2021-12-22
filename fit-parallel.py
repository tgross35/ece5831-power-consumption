"""Run multiple fits in parallel

Framework taken from here https://fredrikaverpil.github.io/2017/06/20/async-and-await-with-subprocesses/"""
import asyncio
import os
import platform
import subprocess
import sys
import time
from glob import glob
from os import wait
from pathlib import Path

import click
from termcolor import colored

from crystalball.constants import RUN_CONSTANTS

CONCURRENCY = 8
MODELS_DIRECTORY = None
RUN_NAME = None
LOG_FIT_DIRECTORY = None


def config_constants():
    runname = os.environ["CRYSTAL_MODEL_NAME"]
    constants = RUN_CONSTANTS[runname]
    global MODELS_DIRECTORY, RUN_NAME, LOG_FIT_DIRECTORY
    RUN_NAME = runname
    MODELS_DIRECTORY = constants["MODELS_DIRECTORY"]
    LOG_FIT_DIRECTORY = constants["LOG_FIT_DIRECTORY"]


async def run_command(command):
    """Run command in subprocess.

    Example from:
        http://asyncio.readthedocs.io/en/latest/subprocess.html
    """
    cmd = command["cmd"]
    fname = command["fname"]
    # Create subprocess
    path = Path(fname)
    if not path.is_file():
        f = open(path, "w")
    else:
        f = open(path, "a")
    process = await asyncio.create_subprocess_exec(*cmd, stdout=f, stderr=f)
    # import pdb

    # pdb.set_trace()
    # for line in process.stdout:  # b'\n'-separated lines
    #     sys.stdout.buffer.write(line)  # pass bytes as is
    #     f.write(line)
    f.close()

    # Status
    print(colored("Started: %s, pid=%s" % (cmd, process.pid), "green"), flush=True)

    # Wait for the subprocess to finish
    stdout, stderr = await process.communicate()

    if stderr is not None:
        f.write(stderr.decode().strip())
    f.close()

    # Progress
    if process.returncode == 0:
        print(
            colored("Done: %s, pid=%s" % (cmd, process.pid), "green"),
            flush=True,
        )
    else:
        print(
            colored(
                "Failed: %s, pid=%s" % (cmd, process.pid),
                "red",
            ),
            flush=True,
        )

    # Result
    # result = stdout.decode().strip()

    # # Return stdout
    # return result


async def run_command_shell(command):
    """Run command in subprocess (shell).

    Note:
        This can be used if you wish to execute e.g. "copy"
        on Windows, which can only be executed in the shell.
    """
    # Create subprocess
    process = await asyncio.create_subprocess_shell(
        command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    # Status
    print("Started:", command, "(pid = " + str(process.pid) + ")", flush=True)

    # Wait for the subprocess to finish
    stdout, stderr = await process.communicate()

    # Progress
    if process.returncode == 0:
        print("Done:", command, "(pid = " + str(process.pid) + ")", flush=True)
    else:
        print("Failed:", command, "(pid = " + str(process.pid) + ")", flush=True)

    # Result
    result = stdout.decode().strip()

    # Return stdout
    return result


def make_chunks(l, n):
    """Yield successive n-sized chunks from l.

    Note:
        Taken from https://stackoverflow.com/a/312464
    """
    for i in range(0, len(l), n):
        yield l[i : i + n]


def run_asyncio_commands(tasks, max_concurrent_tasks=0):
    """Run tasks asynchronously using asyncio and return results.

    If max_concurrent_tasks are set to 0, no limit is applied.

    Note:
        By default, Windows uses SelectorEventLoop, which does not support
        subprocesses. Therefore ProactorEventLoop is used on Windows.
        https://docs.python.org/3/library/asyncio-eventloops.html#windows
    """
    all_results = []

    if max_concurrent_tasks == 0:
        chunks = [tasks]
        num_chunks = len(chunks)
    else:
        chunks = make_chunks(l=tasks, n=max_concurrent_tasks)
        num_chunks = len(list(make_chunks(l=tasks, n=max_concurrent_tasks)))

    if asyncio.get_event_loop().is_closed():
        asyncio.set_event_loop(asyncio.new_event_loop())
    if platform.system() == "Windows":
        asyncio.set_event_loop(asyncio.ProactorEventLoop())
    loop = asyncio.get_event_loop()

    chunk = 1
    for tasks_in_chunk in chunks:
        print("Beginning work on chunk %s/%s" % (chunk, num_chunks), flush=True)
        commands = asyncio.gather(*tasks_in_chunk)  # Unpack list using *
        results = loop.run_until_complete(commands)
        all_results += results
        print("Completed work on chunk %s/%s" % (chunk, num_chunks), flush=True)
        chunk += 1

    loop.close()
    return all_results


def create_command_list():

    globs = glob(f"{MODELS_DIRECTORY}/*")
    modelnames = [g.split("/")[-1] for g in globs if "manifset.txt" not in g]

    cmd_list = []
    for model in modelnames:
        print(colored(f"Adding task {model}", "green"))
        cmd = [
            "python",
            "runcrystal.py",
            "--train",
            model,
            "--fit-only",
            "--runname",
            RUN_NAME,
        ]
        fname = f"{LOG_FIT_DIRECTORY}/{model}.txt"
        cmd_list.append({"cmd": cmd, "fname": fname})
        print(colored(f"Added task {model}", "green"))

    return cmd_list


def main():
    """Main program."""
    start = time.time()

    # Commands to be executed on Unix
    commands = create_command_list()

    tasks = []
    for command in commands:
        tasks.append(run_command(command))

    # # Shell execution example
    # tasks = [run_command_shell('copy c:/somefile d:/new_file')]

    results = run_asyncio_commands(tasks, max_concurrent_tasks=10)

    end = time.time()
    rounded_end = "{0:.4f}".format(round(end - start, 4))
    print("Script ran in about %s seconds" % (rounded_end), flush=True)


@click.option(
    "--runname",
    default=None,
    help="Set run number",
)
@click.command()
def runner(runname):
    os.environ["CRYSTAL_MODEL_NAME"] = runname
    config_constants()
    main()


if __name__ == "__main__":
    runner()
