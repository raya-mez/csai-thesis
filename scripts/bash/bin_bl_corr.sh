#!/bin/bash
#SBATCH -p GPU # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -t 0-36:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --gres=gpu:1

source activate thesis

cd /home/u286379/csai_thesis/

version=3
vocab_path="data/vocab.pkl"
lsa_model_path="models/wiki_lsi_model.model"
output_folder="results/correlations/bin_baseline"

echo "Started computing binned baseline correlations"
python scripts/bin_baseline_$version/bin_baseline.py $vocab_path $lsa_model_path $output_folder
echo "Finished computing binned baseline correlations"