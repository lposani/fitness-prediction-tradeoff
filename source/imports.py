import numpy as np
import os
import matplotlib as mpl
mpl.rcParams.update({'figure.autolayout': True})
import matplotlib.pyplot as plt
pltcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
import shutil
import random
import scipy
import scipy.stats
from scipy.optimize import newton
import pickle
import subprocess
import shlex
import threading
import signal
import time
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind as ttest
from scipy.stats import ttest_rel as ttest_paired
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.stats import wilcoxon as wilcoxon
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import sem

import warnings
warnings.filterwarnings("ignore")

params = {
    'overwrite': False,
    'focusing_overwrite': 0,
    'reweighting': 0.2,
    'variance_cutoff': 0.5,
    'scaling_niter': 5,
    'max_variance_scaling': {'IND': 1.0, 'ACEK': 1.5, 'ACEK0': 1.0},
    'unseen_pseudocount': 0.0,
    'dmin': 0.4,
    'dmax': 0.8,
    'focusing_min_B_IND': 20,
    'focusing_min_B_ACEK': 20,
    'focusing_variance_cutoff': 0.5,
    'focusing_snr_cutoff': 3.0,
    'debug': False,
    'timing': False,
    'contacts': 'F'
}
