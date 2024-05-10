#!/bin/bash
#SBATCH -p GPU # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -t 0-36:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --gres=gpu:1

source activate thesis

cd /home/u286379/csai_thesis/

echo "Started computing cosine and edit distances per word length with 4 rescaling methods"
python scripts/distances_1/compute_distances.py 
echo "Finished computing cosine and edit distances per word length with 4 rescaling methods"