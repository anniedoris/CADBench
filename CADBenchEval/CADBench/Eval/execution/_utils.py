import os
import tempfile
import asyncio
from multiprocessing import Process, Queue
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm.auto import tqdm
import numpy as np
import tabulate
from contextlib import redirect_stdout, redirect_stderr
import sys
from contextlib import contextmanager
import json
import trimesh
import multiprocessing as mp

try:
    import cadquery
except ImportError:
    print("WARNING: cadquery is not installed. All shape evaluations will fail. Install it with: pip install cadquery", flush=True)

TEST_TEMPLATE = """
{code}

import cadquery as cq
import tempfile
import trimesh

temp_stl = tempfile.NamedTemporaryFile(delete=True, suffix=".stl")
{variable_name}.export(temp_stl.name)
mesh = trimesh.load(temp_stl.name)
temp_stl.close()
"""

@contextmanager
def suppress_output_os():
    if sys.platform == "win32":
        yield
        return
    stdout_fd, stderr_fd = 1, 2
    saved_stdout_fd, saved_stderr_fd = os.dup(stdout_fd), os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_RDWR)
    try:
        os.dup2(devnull_fd, stdout_fd)
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)

@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull), redirect_stderr(fnull):
            yield

def __get_mesh(code: str, variable_name: str = "solid") -> trimesh.Trimesh:
    with suppress_output_os(), suppress_output():
        local_vars = {}
        exec_globals = {'__builtins__': __builtins__}
        exec(TEST_TEMPLATE.format(code=code, variable_name=variable_name), exec_globals, local_vars)
        mesh = local_vars.get('mesh')
        return mesh

def _get_mesh(code: str, queue: Queue, variable_name: str = "solid") -> trimesh.Trimesh:
    with suppress_output_os(), suppress_output():
        try:
            mesh = __get_mesh(code, variable_name)
        except Exception as e:
            queue.put((None, 0, f"Error generating mesh: {e}"))
            return

        queue.put((mesh, 1, "Mesh generated successfully"))
        return

def get_mesh_safe(code: str, variable_name: str = "solid") -> trimesh.Trimesh:
    return __get_mesh(code, variable_name)