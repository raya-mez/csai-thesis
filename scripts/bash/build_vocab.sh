#!/bin/bash
#SBATCH -p GPU # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -t 0-36:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --gres=gpu:1

source activate thesis

WORDIDS="data/wiki_wordids.txt.bz2"
BOW_CORPUS="data/wiki_bow.mm"
OUTPUT_FILE="data/vocab.pkl"

echo "Started creating vocabulary"
python scripts/build_vocab.py $WORDIDS $BOW_CORPUS $OUTPUT_FILE
echo "Finished creating vocabulary. Find it in data/vocab.pkl."