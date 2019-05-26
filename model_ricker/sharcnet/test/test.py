#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 11:09:11 2019

Compute bootstrapped EWS for the Ricker model going through the Fold bifurcation


@author: Thomas Bury
"""





# Import python libraries
import numpy as np
import pandas as pd
import os
import sys

import ewstools



#---------------------
# Parameters input from terminal
#–----------------------
  
tmax = int(sys.argv[1])  
seed = int(sys.argv[2])
sigma = float(sys.argv[3])
span = float(sys.argv[4])
rw = float(sys.argv[5])
ham_length = int(sys.argv[6])
ham_offset = float(sys.argv[7])
sweep = bool(sys.argv[8])
block_size = int(sys.argv[9])
bs_type = str(sys.argv[10])
n_samples = int(sys.argv[11])

print('''\nRunning Ricker-Fold script with parameters: 
tmax = %d
seed = %d
sigma = %.2f
span = %.2f
rw = %.2f
ham_length = %d
ham_offset = %.2f
sweep = %s
block_size = %d
bs_type = %s
n_samples = %d\n
'''
% (tmax, seed, sigma, span, rw, ham_length, ham_offset, sweep, block_size, bs_type, n_samples)
)

# Export something








