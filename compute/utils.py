import logging
from typing import List
import subprocess
from multiprocessing.pool import ThreadPool

import torch

class DotDict(dict):
    """ A dictionary class that supports dot notation for set, get and del. Also works recursively. """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, "keys"):
                value = DotDict(value)
            self[key] = value

logging_setup = False

def setup_logging(log_file=None):
    # Ensure we don't do this twice
    global logging_setup
    if logging_setup: return
    logging_setup = True

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s](%(levelname)s) %(message)s", datefmt="%H:%M:%S")

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


def run_cmd(s, **kwargs):
    print(s)
    r = subprocess.run(s, shell=True, **kwargs)
    if r.returncode != 0:
        exit(1)
    return r

def run_cmds(cmds: List[str], num_threads=0, **kwargs):
    closures = map(lambda cmd: (lambda: run_cmd(cmd, **kwargs)), cmds)
    if num_threads == 0:
        for fn in closures:
            fn()
    else:
        tp = ThreadPool(num_threads)
        for fn in closures:
            tp.apply_async(fn)
        tp.close()
        tp.join()

def gaussian_kernel(ksize: int, sigma: float):
    x = torch.arange(0, ksize) - ksize // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma * sigma))
    return gauss / gauss.sum()

def apply_featurewise_conv1d(signal: torch.Tensor, kernel: torch.Tensor, pad_mode="replicate") -> torch.Tensor:
    """ Apply a convolution filter to a 1D signal with multiple feature dimensions (or channels). """
    # signal: (N, n_features)
    # kernel: (kernel_size)
    _, n_features = signal.shape
    kernel_size = kernel.shape[0]
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(n_features,1,1) # (n_features,1,kernel_size) (we want as many output channels as features)
    signal = signal.permute(1,0).unsqueeze(0) # (1, n_features, N)
    # Pad input signal
    padding = kernel_size // 2 # to maintain the original signal length
    padded_signal = torch.nn.functional.pad(signal, (padding, padding), mode=pad_mode)
    # Perform convolution
    filtered_signal = torch.nn.functional.conv1d(padded_signal, kernel, groups=n_features) # (1, n_features, N)
    return filtered_signal.squeeze(0).permute(1,0) # (N, n_features)
