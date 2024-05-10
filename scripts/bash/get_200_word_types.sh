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
OUTPUT_FILE_PATH="data/selected_wordtypes.txt"

echo "Started extracting 200 word types"
python scripts/get_200_word_types.py $WORDIDS $BOW_CORPUS $OUTPUT_FILE_PATH
echo "Finished extracting 200 word types. Find them in data/selected_wordtypes.txt"
