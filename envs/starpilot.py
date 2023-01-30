import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import gymnasium as gym
import procgen
import random
import copy
from collections import deque

# need to fix these imports
from procgen import ProcgenEnv
from vec_env import VecExtractDictObs
from vec_env import VecMonitor
from vec_env import VecNormalize
from util import logger

from ppo import *
from policies import *


def main(): # parse a config file for parameters
    