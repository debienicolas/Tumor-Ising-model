"""
Full run of the ising model on the branching network.
Using wand to log runs and results. 

Calculate graph statistics and also save them to wandb.

"""

import wandb
import os
import sys
import time

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from IsingModel import IsingModel
from main_tree import simulate_ising_model



