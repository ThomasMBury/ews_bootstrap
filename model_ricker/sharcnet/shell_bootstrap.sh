#!/bin/bash

# Make sure python 3 is activated in conda
source activate py3

# Shell to run python scripts that compute bootstrap EWS of Ricker model

block_sizes=($(seq 20 10 25))
roll_windows=($(seq 0.2 0.1 0.3))



# Loop over parameters and run scripts
for block_size in "${block_sizes[@]}"; do
   for rw in "${roll_windows[@]}"; do
	   python script_flip_gen.py $block_size $rw
	   python script_fold_gen.py $block_size $rw
   done 
done


