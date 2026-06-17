from ..Processing import Processor
from typing import Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
import multiprocessing as mp
import os
import sys
from contextlib import contextmanager, redirect_stdout, redirect_stderr


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
    with open(os.devnull, "w") as fnull:
        with redirect_stdout(fnull), redirect_stderr(fnull):
            yield

class InferenceEngine:
    def __init__(self,
                 processor: Processor,
                 num_workers: int = 16,
                 verbose: bool = True):
        self.processor = processor
        self.num_workers = num_workers
        self.verbose = verbose

    def _run_inference(self, input_data: Any):
        pass

    def _process_item(self, idx: int):
        with suppress_output_os(), suppress_output():
            input_data, file_id = self.processor[idx]
            output = self._run_inference(input_data)
        return {"file_id": file_id, **output}

    def run(self):
        with ProcessPoolExecutor(max_workers=self.num_workers, mp_context=mp.get_context('fork')) as executor:
            futures = {executor.submit(self._process_item, idx): idx for idx in range(len(self.processor))}
            results = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="Running inference", disable=not self.verbose):
                results.append(future.result())
        return results
