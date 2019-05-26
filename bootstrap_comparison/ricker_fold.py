#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 11:09:11 2019

Compute EWS without bootstrapping.
Ricker model going through the Fold bifurcation.


@author: Thomas Bury
"""




# Import python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


from ewstools import ewstools

# Name of directory within data_export
dir_name = 'tmax200_rw04'

if not os.path.exists('data_export/'+dir_name):
    os.makedirs('data_export/'+dir_name)


# Print update
print('''Compute EWS for multiple simulations of the Ricker model 
         going through the Fold bifurcation''')


#--------------------------------
# Global parameters
#–-----------------------------


# Simulation parameters
dt = 1 # time-step (must be 1 since discrete-time system)
t0 = 0
tmax = 200
tburn = 100 # burn-in period
numSims = 2
seed = 1 # random number generation seed
sigma = 0.02 # noise intensity

# EWS parameters
dt2 = 1 # spacing between time-series for EWS computation
span = 0.5
rw = 0.4
ews = ['var','ac','cv','skew']
lags = [1] # autocorrelation lag times
ham_length = 80 # number of data points in Hamming window
ham_offset = 0.5 # proportion of Hamming window to offset by upon each iteration
pspec_roll_offset = 20 # offset for rolling window when doing spectrum metrics
sweep = False # during optimisation, sweep through initialisation parameters


# Bootstrapping parameters
block_size = 20 # size of blocks used to resample time-series
bs_type = 'Stationary' # type of bootstrapping
n_samples = 2 # number of bootstrapping samples to take
roll_offset = 10 # rolling window offset



#----------------------------------
# Simulate many (transient) realisations
#----------------------------------

# Model
    
# Model parameters
r = 0.75 # growth rate
k = 10 # carrying capacity
h = 0.75 # half-saturation constant of harvesting function
bl = 0 # bifurcation parameter (harvesting) low
bh = 2.7 # bifurcation parameter (harvesting) high
bcrit = 2.364 # bifurcation point (computed in Mathematica)
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





# Initialise a list to collect trajectories
list_traj_append = []

# loop over simulations
print('\nBegin simulations \n')
for j in range(numSims):
    
    
    # Create brownian increments (s.d. sqrt(dt))
    dW_burn = np.random.normal(loc=0, scale=sigma*np.sqrt(dt), size = int(tburn/dt))
    dW = np.random.normal(loc=0, scale=sigma*np.sqrt(dt), size = len(t))
    
    # Run burn-in period on x0
    for i in range(int(tburn/dt)):
        x0 = de_fun(x0,r,k,bl,h,dW_burn[i])
        
    # Initial condition post burn-in period
    x[0]=x0
    
    # Run simulation
    for i in range(len(t)-1):
        x[i+1] = de_fun(x[i],r,k,b.iloc[i], h,dW[i])
        # make sure that state variable remains >= 0
        if x[i+1] < 0:
            x[i+1] = 0
            
    # Store series data in a temporary DataFrame
    data = {'Realisation number': (j+1)*np.ones(len(t)),
                'Time': t,
                'x': x}
    df_temp = pd.DataFrame(data)
    # Append to list
    list_traj_append.append(df_temp)
    
    print('Simulation '+str(j+1)+' complete')

#  Concatenate DataFrame from each realisation
df_traj = pd.concat(list_traj_append)
df_traj.set_index(['Realisation number','Time'], inplace=True)






#----------------------
## Compute EWS of each realisation (no bootstrapping)
#---------------------

# Filter time-series to have time-spacing dt2
df_traj_filt = df_traj.loc[::int(dt2/dt)]

# set up a list to store output dataframes from ews_compute- we will concatenate them at the end
appended_ews = []
appended_ktau = []

# loop through realisation number
print('\nBegin EWS computation\n')
for i in range(numSims):
    # loop through variable
    for var in ['x']:
        
        ews_dic = ewstools.ews_compute(df_traj_filt.loc[i+1][var], 
                          roll_window = rw, 
                          span = span,
                          lag_times = lags, 
                          ews = ews,
                          upto = tcrit)
        
        # The DataFrame of EWS
        df_ews_temp = ews_dic['EWS metrics']

        # The DataFrame of ktau values
        df_ktau_temp = ews_dic['Kendall tau']
        
        # Include a column in the DataFrames for realisation number and variable
        df_ews_temp['Realisation number'] = i+1
        df_ews_temp['Variable'] = var

        
        df_ktau_temp['Realisation number'] = i+1
        df_ktau_temp['Variable'] = var
                
        # Add DataFrames to list
        appended_ews.append(df_ews_temp)
#        appended_pspec.append(df_pspec_temp)
        appended_ktau.append(df_ktau_temp)
        
    # Print status every realisation
    if np.remainder(i+1,1)==0:
        print('EWS for realisation '+str(i+1)+' complete')


# Concatenate EWS DataFrames. Index [Realisation number, Variable, Time]
df_ews = pd.concat(appended_ews).reset_index().set_index(['Realisation number','Variable','Time'])


# Concatenate kendall tau DataFrames. Index [Realisation number, Variable]
df_ktau = pd.concat(appended_ktau).reset_index().set_index(['Realisation number','Variable'])



#----------------------
## Compute EWS of each realisation *with bootstrapping*
#---------------------

print('\nBegin bootstrapped EWS computation\n')


# Make list of samples from each realisation
samples_list = []

for i in range(numSims):
    
    # Compute samples for particular realisation
    df_samples_temp = ewstools.roll_bootstrap(df_traj.loc[i+1]['x'],
                   span = span,
                   roll_window = rw,
                   roll_offset = roll_offset,
                   upto = tcrit,
                   n_samples = n_samples,
                   bs_type = bs_type,
                   block_size = block_size
                   )
    
    # Add realisation number
    df_samples_temp['Realisation number'] = i+1
    
    # Add dataframe to list
    samples_list.append(df_samples_temp)

# Concatenate into a full DataFrame of sample time-series
df_samples = pd.concat(samples_list).reset_index().set_index(['Realisation number','Time','Sample','Wintime'])



# Execute ews_compute for each bootstrapped time-series


# List to store EWS DataFrames
list_df_ews_boot = []

# Realtime values
tVals = np.array(df_samples.index.levels[1])
# Sample values
sampleVals = np.array(df_samples.index.levels[2])

# Loop through realisation number
for i in range(numSims):
    
    # Loop through realtimes
    for t in tVals:
        
        # Loop through sample values
        for sample in sampleVals:
            
            # Compute EWS for near-stationary sample series
            series_temp = df_samples.loc[i+1].loc[t].loc[sample]['x']
            
            ews_dic = ewstools.ews_compute(series_temp,
                              roll_window = 1, #Already within a rw
                              smooth = 'None',
                              ews = ews,
                              lag_times = lags,
                              upto='Full')
            
            # The DataFrame of EWS
            df_ews_boot_temp = ews_dic['EWS metrics']
                
            # Drop NaN values
            df_ews_boot_temp.dropna(inplace=True)     
            
            # Include columns for realnum, sample value and realtime
            df_ews_boot_temp['Realisation number'] = i+1
            df_ews_boot_temp['Sample'] = sample
            df_ews_boot_temp['Time'] = t
            
            # Append list_df_ews
            list_df_ews_boot.append(df_ews_boot_temp)

        
    # Print update
    print('EWS for realisation %d complete' %(i+1))
        
# Concatenate EWS DataFrames. Index [Realtime, Sample]
df_ews_boot = pd.concat(list_df_ews_boot).reset_index(drop=True).set_index(['Realisation number','Sample','Time']).sort_index()


# Compute mean of EWS over the samples
df_ews_means = df_ews_boot.mean(level=(0,2))

# Compute the kendall tau values for each realisation number
time_vals = df_ews_means.loc[1].index.tolist()

ktau_var = df_ews_means['Variance'].corr(time_vals, method='kendall')
ktau_ac1 = df_ews_means['Lag-1 AC'].corr(time_vals, method='kendall')















#-------------------------
# Plots to visualise EWS
#-------------------------

# Realisation number to plot
plot_num = 1
var = 'x'
## Plot of trajectory, smoothing and EWS of var (x or y)
fig1, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(6,3))
df_ews.loc[plot_num,var][['State variable','Smoothing']].plot(ax=axes[0],
          title='Early warning signals for a single realisation')
df_ews.loc[plot_num,var]['Variance'].plot(ax=axes[1],legend=True)
df_ews.loc[plot_num,var][['Lag-1 AC']].plot(ax=axes[1], secondary_y=True,legend=True)





# Box plot of Kendall values

fig2 = df_ktau[['Variance','Coefficient of variation','Lag-1 AC']].plot(kind='box')
fig2.set_ylim(0,1)



#------------------------------------
## Export data / figures
#-----------------------------------


#
## Export kendall tau values
#df_ktau.to_csv('data_export/'+dir_name+'/ktau.csv')












