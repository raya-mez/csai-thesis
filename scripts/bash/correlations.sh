#!/bin/bash
#SBATCH -p GPU # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -t 0-36:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --gres=gpu:1

source activate thesis

# Directory containing the CSV files with cosine distance and edit distance scores
DIRECTORY="results/cos_edit_dist"
CORR_SCORES_FILE="results/correlations/pearson_correlations.csv"
mkdir results/correlations/corr_plots
PLOTS_DIRECTORY="results/correlations/corr_plots"

echo "Started calculating Pearson correlations"
python scripts/correlations_per_length.py $DIRECTORY $CORR_SCORES_FILE $PLOTS_DIRECTORY
echo "Done calculating and plotting correlations"