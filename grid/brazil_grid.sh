#!/bin/bash

source /cvmfs/icarus.opensciencegrid.org/products/icarus/setup_icarus.sh
setup icaruscode v09_89_01_02 -qe26:prof

cp $INPUT_TAR_DIR_LOCAL/grid_dir/* $_CONDOR_SCRATCH_DIR/
cd $_CONDOR_SCRATCH_DIR

PROfit -x pandora_mc.xml -t pandora -o brazil_${PROCESS} -s 1$PROCESS -c PROCNP -v 2 --log profit_${PROCESS}.log surface -g 60 --xlo 1e-2 --brazil-band --only-throw --single-throw

if [ -e pandora_brazil_${PROCESS}_surf.root ]
then
  ifdh cp pandora_brazil_${PROCESS}_surf.root /pnfs/icarus/scratch/users/jlarkin/pandora_brazil_again/pandora_brazil_${PROCESS}_surf.root
fi

#if [ -e profit_${PROCESS}.log ]
#then
#  ifdh cp profit_${PROCESS}.log /pnfs/icarus/scratch/users/jlarkin/logs_pandora/profit_${PROCESS}.log
#fi

#PROfit -x spine_mc.xml -t spine -o brazil_${PROCESS} -s 2${PROCESS} -c PROCNP -v 2 --log spine_profit_${PROCESS}.log -n 4 surface -g 60 --xlo 1e-2 --brazil-band --only-throw --single-throw

#if [ -e spine_brazil_${PROCESS}_surf.root ]
#then
#  ifdh cp spine_brazil_${PROCESS}_surf.root /pnfs/icarus/scratch/users/jlarkin/spine_brazil_4k/spine_brazil_${PROCESS}_surf.root
#fi
#
#if [ -e spine_profit_${PROCESS}.log ]
#then
#  ifdh cp spine_profit_${PROCESS}.log /pnfs/icarus/scratch/users/jlarkin/logs_spine/spine_profit_${PROCESS}.log
#fi

