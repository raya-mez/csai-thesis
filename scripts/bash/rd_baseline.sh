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
mkdir results/rd_baseline_${version}
output_file="results/rd_baseline_${version}/rd_bl_corr.csv"

echo "Started computing random baseline correlation"
python scripts/rd_baseline_$version/rd_baseline.py $vocab_path $lsa_model_path $output_file
echo "Finished computing random baseline correlation"