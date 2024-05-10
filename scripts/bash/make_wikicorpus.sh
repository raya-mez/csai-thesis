#!/bin/bash
#SBATCH -p GPU # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -t 0-36:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --gres=gpu:1

source activate thesis

echo "Started make_wikicorpus"
python -m gensim.scripts.make_wikicorpus data/enwiki-latest-pages-articles.xml.bz2 data/wiki 10000
echo "Done make_wikicorpus"