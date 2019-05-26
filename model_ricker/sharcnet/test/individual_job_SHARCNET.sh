#!/bin/bash

# # Thomas Bury
# # PhD Candidate
# # Bauch computational epidemiology research group
# # Department of Applied Mathematics
# # Faculty of Mathematics
# # University of Waterloo

#SBATCH --mem=250MB
#SBATCH --time=0-00:01:00
#SBATCH --output=Jobs/job.%N.%j.out
#SBATCH --ntasks=1

. test.py `cat job_table.txt | head -n $1 | tail -n 1`
