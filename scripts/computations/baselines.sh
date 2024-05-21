#!/bin/bash
#SBATCH -p GPU # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -t 0-36:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --gres=gpu:1

source activate thesis

cd /home/u286379/csai_thesis/

# Define vocabularies
vocabs=("vocab_raw" "vocab_monomorph")

# Loop over each vocabulary in the list
for vocab in "${vocabs[@]}"; do  
    echo "Processing vocabulary: $vocab"
    # Execute the Python script with the current vocabulary
    # python scripts/computations/compute_rd_baseline_correlations.py "$vocab"  
    python scripts/computations/compute_bin_baseline_correlations.py "$vocab"  
    echo "Finished computing for vocabulary: $vocab"
done

echo "Finished computing binned baseline correlations"