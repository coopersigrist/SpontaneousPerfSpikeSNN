import argparse
import os
from time import time as t

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from bindsnet import ROOT_DIR
from bindsnet.analysis.plotting import (
    plot_assignments,
    plot_input,
    plot_performance,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))


inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
assigns_im = None
perf_ax = None

step = 27520

after = torch.load("../saved models/"+str(n_neurons)+"_"+str(step)+"_after")
before = torch.load("../saved models/"+str(n_neurons)+"_before")

diff = (after.connections[("X", "Ae")].w - before.connections[("X", "Ae")].w).cpu()
square_weights = get_square_weights(
diff.view(784, n_neurons), n_sqrt, 28)
print(np.sum(diff.numpy()))
weights_im = plot_weights(square_weights, im=weights_im)
wait = input("done?")