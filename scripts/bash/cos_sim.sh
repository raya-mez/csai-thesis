#!/bin/bash
#SBATCH -p GPU # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -t 0-36:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --gres=gpu:1

source activate thesis

VOCAB="data/vocab.pkl"
LSA_MODEL="models/wiki_lsi_model.model"
OUTPUT_FILE_PATH="results/cos_sim.csv"

echo "Started computing cosine similarities"
python scripts/cos_sim.py $VOCAB $LSA_MODEL $OUTPUT_FILE_PATH
echo "Finished computing cosine similarities. Find the results in results/cos_sim.csv"