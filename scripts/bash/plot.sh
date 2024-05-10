#!/bin/bash
#SBATCH -p GPU # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -t 0-36:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --gres=gpu:1

source activate thesis
cd /home/u286379/csai_thesis/

INPUT_CSV="results/avg_dist.csv" 
OUTPUT_IMAGE="plots/avg_dist_scatterplot.png"

python scripts/plotting/plot_main.py $INPUT_CSV $OUTPUT_IMAGE