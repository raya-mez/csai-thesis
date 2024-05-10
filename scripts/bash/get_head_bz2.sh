#!/bin/bash
#SBATCH -p GPU # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -t 0-36:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --gres=gpu:1

source activate thesis

INPUT_FILE_PATH="data/wiki_wordids.txt.bz2"
OUTPUT_FILE_PATH="data/head_wiki_wordids.txt"

N_LINES=1000

python scripts/get_head_bz2.py "$INPUT_FILE_PATH" "$OUTPUT_FILE_PATH" $N_LINES
