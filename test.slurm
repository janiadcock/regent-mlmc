#!/bin/bash

#SBATCH -J test               # Job name
#SBATCH -o job.%j.out         # Name of stdout output file (%j expands to jobId)
#SBATCH -N 2                  # Total number of nodes requested
#SBATCH --mincpus 3           # Minimum number of cpus per node
#SBATCH -n 6                  # Total number of mpi tasks requested
#SBATCH -t 01:30:00           # Run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-type=ALL       # mail alert
#SBATCH --mail-user=janiad@stanford.edu # replace [sunetid] with your SUNetID

echo "Running [your app] with: $cmd on $SLURM_JOB_NODELIST in directory "`pwd`

