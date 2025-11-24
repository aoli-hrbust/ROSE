# MIT License

# Copyright (c) 2025 Ao Li

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam


def get_device():
    return 0 if torch.cuda.is_available() else 'cpu'

def convert_tensor(thing, dtype=torch.float, dev="cpu"):
    """
    Convert a np.ndarray or list of them to tensor.
    """
    if isinstance(thing, (list, tuple)):
        return [convert_tensor(x, dtype, dev) for x in thing]
    elif isinstance(thing, dict):
        return {key: convert_tensor(val) for key, val in thing.items()}
    elif isinstance(thing, np.ndarray):
        return torch.tensor(thing, dtype=dtype, device=dev)
    elif isinstance(thing, torch.Tensor):
        return thing
    elif thing is None:
        return None
    else:
        raise ValueError(f"{type(thing)}")


def convert_numpy(thing):
    """
    Convert a tensor or list of them to numpy.
    """
    if isinstance(thing, (list, tuple)):
        return [convert_numpy(x) for x in thing]
    elif isinstance(thing, dict):
        return {key: convert_numpy(val) for key, val in thing.items()}
    elif isinstance(thing, torch.Tensor):
        return thing.detach().cpu().numpy()
    else:
        return thing


def convert_cpu(thing):
    """
    Convert a tensor or list of them to numpy.
    """
    if isinstance(thing, (list, tuple)):
        return [convert_cpu(x) for x in thing]
    elif isinstance(thing, dict):
        return {key: convert_cpu(val) for key, val in thing.items()}
    elif isinstance(thing, torch.Tensor):
        return thing.detach().cpu()
    else:
        return thing

