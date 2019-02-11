#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 11:09:11 2019

Compute bootstrapped EWS for the Ricker model going through the Flip bifurcation


@author: Thomas Bury
"""




# Import python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os



# Import EWS module
import sys
sys.path.append('../../early_warnings')
from ews_compute import ews_compute

# Import bootstrap module
sys.path.append('../')
from roll_bootstrap import roll_bootstrap


# Print update
print("Compute bootstrapped EWS for the Ricker model going through the Flip bifurcation")

#--------------------------------
# Global parameters
#–-----------------------------


# Simulation parameters
dt = 1 # time-step (must be 1 since discrete-time system)
t0 = 0
tmax = 1000
tburn = 100 # burn-in period
seed = 2 # random number generation seedaa
sigma = 0.02 # noise intensity



# EWS parameters
span = 0.5
rw = 0.25
ews = ['var','ac','smax','aic']
lags = [1,2,3] # autocorrelation lag times
ham_length = 40 # number of data points in Hamming window
ham_offset = 0.5 # proportion of Hamming window to offset by upon each iteration
pspec_roll_offset = 20 # offset for rolling window when doing spectrum metrics


# Bootstrapping parameters
block_size = 10 # size of blocks used to resample time-series
bs_type = 'Stationary' # type of bootstrapping
n_samples = 100 # number of bootstrapping samples to take
roll_offset = 10 # rolling window offset




#----------------------------------
# Simulate transient realisation of Ricker model
#----------------------------------

    
# Model parameters
f = 0 # harvesting rate (fixed for exploring flip bifurcation)
k = 10 # carrying capacity
h = 0.75 # half-saturation constant of harvesting function
bl = 0.5 # bifurcation parameter (growth rate) low
bh = 2.6 # bifurcation parameter (growth rate) high
bcrit = 2 # bifurcation point (computed in Mathematica)
x0 = 0.8 # initial condition

def de_fun(x,r,k,f,h,xi):
    return x*np.exp(r*(1-x/k)+xi) - f*x**2/(x**2+h**2)


# Initialise arrays to store single time-series data
t = np.arange(t0,tmax,dt)
x = np.zeros(len(t))

# Set bifurcation parameter b, that increases linearly in time from bl to bh
b = pd.Series(np.linspace(bl,bh,len(t)),index=t)
# Time at which bifurcation occurs
tcrit = b[b > bcrit].index[1]

## Implement Euler Maryuyama for stocahstic simulation

# Set seed
np.random.seed(seed)
   
# Create brownian increments (s.d. sqrt(dt))
dW_burn = np.random.normal(loc=0, scale=sigma*np.sqrt(dt), size = int(tburn/dt))
dW = np.random.normal(loc=0, scale=sigma*np.sqrt(dt), size = len(t))

# Run burn-in period on x0
for i in range(int(tburn/dt)):
    x0 = de_fun(x0,bl,k,f,h,dW_burn[i])
    
# Initial condition post burn-in period
x[0]=x0

# Run simulation
for i in range(len(t)-1):
    x[i+1] = de_fun(x[i],b.iloc[i],k,f,h,dW[i])
    # make sure that state variable remains >= 0
    if x[i+1] < 0:
        x[i+1] = 0
        
# Trajectory data stored in a DataFrame indexed by time
data = { 'Time': t,
            'x': x}
df_traj = pd.DataFrame(data).set_index('Time')





#--------------------------------
# Compute EWS (moments) without bootstrapping
#-------------------------------------

# Time-series data as a pandas Series
series = df_traj['x']
        
# Put into ews_compute
ews_dic = ews_compute(series,
                      smooth = 'Lowess',
                      span = span,
                      roll_window = rw,
                      upto = tcrit,
                      ews = ews,
                      lag_times = lags)

# DataFrame of EWS
df_ews = ews_dic['EWS metrics']

# Plot trajectory and smoothing
df_ews[['State variable','Smoothing']].plot()

# Plot variance
df_ews[['Variance']].plot()






#-------------------------------------
# Compute EWS using bootstrapping
#–----------------------------------

df_samples = roll_bootstrap(series,
                   span = span,
                   roll_window = rw,
                   roll_offset = roll_offset,
                   upto = tcrit,
                   n_samples = n_samples,
                   bs_type = bs_type,
                   block_size = block_size
                   )

# Execute ews_compute for each bootstrapped time-series


# List to store EWS DataFrames
list_df_ews = []

# Realtime values
tVals = np.array(df_samples.index.levels[0])
# Sample values
sampleVals = np.array(df_samples.index.levels[1])



# Loop through realtimes
for t in tVals:
    
    # Loop through sample values
    for sample in sampleVals:
        
        # Compute EWS for near-stationary sample series
        series_temp = df_samples.loc[t].loc[sample]['x']
        
        ews_dic = ews_compute(series_temp,
                          roll_window = 1, 
                          band_width = 1,
                          ews = ews,
                          lag_times = lags,
                          upto='Full')
        
        # The DataFrame of EWS
        df_ews_temp = ews_dic['EWS metrics']
        
        # Include columns for sample value and realtime
        df_ews_temp['Sample'] = sample
        df_ews_temp['Time'] = t

        # Drop NaN values
        df_ews_temp = df_ews_temp.dropna()        
        
        # Append list_df_ews
        list_df_ews.append(df_ews_temp)
    
    # Print update
    print('EWS for t=%.2f complete' % t)
        
# Concatenate EWS DataFrames. Index [Realtime, Sample]
df_ews_boot = pd.concat(list_df_ews).reset_index(drop=True).set_index(['Time','Sample'])








#--------------------------------------
# Plot summary statistics of EWS
#--------------------------------------


## Plot of variance of bootstrapped samples
# Put DataFrame in form for Seaborn plot
data = df_ews_boot.reset_index().melt(id_vars = 'Time',
                         value_vars = ('Variance'),
                         var_name = 'EWS',
                         value_name = 'Magnitude')
# Make plot with error bars
var_plot = sns.relplot(x="Time", 
            y="Magnitude",
            hue="EWS", 
            kind="line", 
            data=data)



## Plot of autocorrelation of bootstrapped samples
# Put DataFrame in form for Seaborn plot
data = df_ews_boot.reset_index().melt(id_vars = 'Time',
                         value_vars = ('Lag-1 AC', 'Lag-2 AC','Lag-3 AC'),
                         var_name = 'EWS',
                         value_name = 'Magnitude')
# Make plot with error bars
ac_plot = sns.relplot(x="Time", 
            y="Magnitude",
            hue="EWS", 
            kind="line", 
            data=data)






## Plot of Smax of bootstrapped samples
# Put DataFrame in form for Seaborn plot
data = df_ews_boot.reset_index().melt(id_vars = 'Time',
                         value_vars = ('Smax'),
                         var_name = 'EWS',
                         value_name = 'Magnitude')
# Make plot with error bars
smax_plot = sns.relplot(x="Time", 
            y="Magnitude",
            hue="EWS", 
            kind="line", 
            data=data)





#----------------------------------
# Compute quantiles of bootstrapped EWS
#–---------------------------------------

# Quantiles to compute
quantiles = [0.05,0.25,0.5,0.75,0.95]

# DataFrame of quantiles for each EWS
df_quant = df_ews_boot.groupby(level=0).quantile(quantiles, axis=0)
# Rename and reorder index of DataFrame
df_quant.index.rename(['Time','Quantile'], inplace=True)
df_quant = df_quant.reorder_levels(['Quantile','Time']).sort_index()

## Plot of quantiles
#df_quant.loc[0.05:0.95]['Variance'].unstack(level=0).plot()




#-------------------------------------
# Export data for plotting in MMA
#–------------------------------------

df_ews.reset_index().to_csv('data_export/ews_flip.csv')

df_quant.reset_index().to_csv('data_export/ews_flip_boot.csv')






