#!/bin/bash

# module load gcc/7.3.0

mkdir -p Jobs
mkdir -p Data
rm Skynet

make

. job_table_generator.sh

MAX=`cat job_table.txt | wc -l`

for i in `seq 2 $MAX`; do
	. individual_job_SHARCNET.sh $i
	sleep 1.0
done
