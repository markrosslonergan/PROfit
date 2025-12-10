#!/bin/bash

source /cvmfs/icarus.opensciencegrid.org/products/icarus/setup_icarus.sh
setup icaruscode v09_89_01_02 -qe26:prof

cp $INPUT_TAR_DIR_LOCAL/grid_dir/* $_CONDOR_SCRATCH_DIR/
cd $_CONDOR_SCRATCH_DIR

# Variable that controls where the output goes
N=71

# For a list of "X Y" points in a file
X=$(awk "NR==$((PROCESS+1))" pandora_5k.txt | awk '{print $1}')
Y=$(awk "NR==$((PROCESS+1))" pandora_5k.txt | awk '{print $2}')
#X=$(awk "NR==$((PROCESS+1))" spine_5k.txt | awk '{print $1}')
#Y=$(awk "NR==$((PROCESS+1))" spine_5k.txt | awk '{print $2}')

# Or a hardcoded X, Y value
#X=21
#Y=59

# The order I happened to generate the values in was looping over DMSQ first then SINSQ2T
# If these files are made in the other order this would have to change
PT=$((60*X+Y))
DM=$(awk "NR==$((PT+1))" pandora_fine.txt | awk '{print $2}')
SST=$(awk "NR==$((PT+1))" pandora_fine.txt | awk '{print $1}')
#DM=$(awk "NR==$((PT+1))" spine_fine.txt | awk '{print $2}')
#SST=$(awk "NR==$((PT+1))" spine_fine.txt | awk '{print $1}')

# For a list of "sinsq2t dmsq" values in a file
#DM=$(awk "NR==$((PROCESS+1))" pandora_fine.txt | awk '{print $2}')
#SST=$(awk "NR==$((PROCESS+1))" pandora_fine.txt | awk '{print $1}')
#DM=$(awk "NR==$((PROCESS+1))" spine_fine.txt | awk '{print $2}')
#SST=$(awk "NR==$((PROCESS+1))" spine_fine.txt | awk '{print $1}')
#PT=$PROCESS

PROfit -x pandora_mc.xml -t pandora -o fc_${PT} -s 19${N} -c PROCNP -v 2 --inject dmsq $DM sinsq2thmm $SST --scan-fit-options max_iterations 100 fc -u 100

if [ -e pandora_fc_${PT}_FC.root ]
then
  ifdh cp pandora_fc_${PT}_FC.root /pnfs/icarus/scratch/users/jlarkin/pandora_fine/pandora_${N}/pandora_fc_${PT}_FC.root
fi

#N=$PROCESS
#PROfit -x spine_mc.xml -t spine -o fc_${PT} -s 190${N} -c PROCNP -v 2 --inject dmsq $DM sinsq2thmm $SST --scan-fit-options max_iterations 100 fc -u 100
#
#if [ -e spine_fc_${PT}_FC.root ]
#then
#  ifdh cp spine_fc_${PT}_FC.root /pnfs/icarus/scratch/users/jlarkin/spine_fine/spine_${N}/spine_fc_${PT}_FC.root
#fi
#
#if [ -e spine_profit_${PT}.log ]
#then
#  ifdh cp spine_profit_${PT}.log /pnfs/icarus/scratch/users/jlarkin/logs_spine_${N}/spine_profit_${PT}.log
#fi

# Using 4 cores requires adding --cpu 4
#PROfit -x pandora_mc.xml -t pandora -o fc_${N} -s 19${N} -c PROCNP -v 2 --inject dmsq $DM sinsq2thmm $SST --scan-fit-options max_iterations 100 -n 4 fc -u 200
#
#if [ -e pandora_fc_${N}_FC.root ]
#then
#  ifdh cp pandora_fc_${N}_FC.root /pnfs/icarus/scratch/users/jlarkin/feld_test/pandora_4/pandora_fc_${N}_FC.root
#fi
