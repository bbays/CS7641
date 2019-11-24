README

#############
CS7641 Machine Learning
Assignment 4
Markov Decision Processes

Author: Bryan Baysinger
Date: November 24, 2019
#############

Code Location
#############
https://github.com/bbays/CS7641

Summary
#############

In this project there are 2 environments.  One from OpenAI is a Frozen Lake, one that is deterministic, and one variety that is stochastic, adding a slippery tile environment.  The Frozen Lake environments are grid worlds tested at various sizes doubling from a 4x4, 8x8, 16x16, and to a 32x32.  The second environment is a forest management environment by Steven A W Cordwell.

With these 2 environments we conduct a series of Markov Decision Process starting with Value policy and value iteration, and leading into value policy and value iteration, and ultimately conduct a Q learning model on the selected environments.

Code and Datasets for Graphs and Tables
#############
https://github.com/bbays/CS7641


Requirements - Latest Version unless specified
#############
import numpy as np
import pandas as pd
import random
from time import time
import itertools 
import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
from gym.envs.registration import register
from gym import wrappers
from hiive.mdptoolbox import mdp
from collections import defaultdict
import sys
from collections import namedtuple
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
%matplotlib inline


Run
#############
1) From Terminal launch Jupyter Notebook at ipynb notebook location
2) First cell includes all libraries, any error here likely a missing library package-ignore deprecation of six
3) Run cells associated with given learner to test
4) Charts and scores presented in line and derived from data frame exports to excel tables



Resources and Citations
#############
Presented in line in the notebook and in the accompanying analysis
