#!/bin/bash

# Usage:
#  mktar_and_submit.sh path/to/grid_script.sh other_file_to_include and_another_file_to_include ...

if [ $# -eq 0 ]; then
  echo "Expected at least the name of the script to submit"
  exit 1
fi

SCRIPT=$1
shift

mkdir grid_dir
cd grid_dir
cp ../../build/bin/PROfit .
cp ../../build/bin/PROfit_dict_rdict.pcm
cp ../$SCRIPT .
cd ..
cp $@ grid_dir/

tar cf grid_dir.tar grid_dir

# There's a long tail of jobs that take >1d to finish, so I request a 2 day lifetime
# I've found I can request up to 3 days before the submission is rejected
jobsub_submit -G icarus --expected-lifetime=2d --lines '+FERMIHTC_AutoRelease=True' --lines '+FERMIHTC_GraceMemory=4000' --lines '+FERMIHTC_GraceLifetime=7200' --role=Analysis --resource-provides="usage_model=DEDICATED,OPPORTUNISTIC,OFFSITE" -l '+SingularityImage=\"/cvmfs/singularity.opensciencegrid.org/fermilab/fnal-wn-sl7:latest\"' --append_condor_requirements='(TARGET.HAS_SINGULARITY=?=true)' --tar_file_name "dropbox://$(pwd)/grid_dir.tar" "file://$(pwd)/${SCRIPT}"
