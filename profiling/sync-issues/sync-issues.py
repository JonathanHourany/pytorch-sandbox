"""
Forcing the Host and Device to synchronize when they otherwise wouldn't have to can be
a real source of inefficiency as processes on both devices must halt and wait for data
to be put on the bus and transmitted. Often, the cause of these syncs can be very
subtle and unintuitive. The easiest place to spot this is calling `.item()` on a tensor
that is in the GPU, but what is often missed is that using a Python object to slice an
object also forces a Host->Device transmission.

@author: Jonathan Hourany
"""

import torch
import torch.nn as nn
import torch.optim as optim
from rich.traceback import install
from torch.profiler import ProfilerActivity, profile, schedule
