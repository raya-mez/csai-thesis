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
OUTPUT_FOLDER="results"

echo "Started computing cosine similarities per word length"
python scripts/cos_dist_wordlength.py $VOCAB $LSA_MODEL $OUTPUT_FOLDER
echo "Finished computing cosine similarities per word length. Find the results in results/"