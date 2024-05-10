#!/bin/bash
#SBATCH -p GPU # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -t 0-36:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --gres=gpu:1

source activate thesis

mkdir results/cos_dist_per_word_length
# Directory containing files with cosine similarity scores
DIRECTORY="results/cos_dist_per_word_length"

# Iterate over files in the directory
for FILE in $DIRECTORY; do
    # Check if the file is a regular file
    if [ -f "$FILE" ]; then
        # Get the filename without extension
        FILENAME=$(basename "$FILE" .csv)
        # Set the output filename with 'edit_dist' appended
        OUTPUT="${DIRECTORY}/${FILENAME}_edit_dist.csv"
        
        echo "Started edit distance script for $FILE"
        # Run the Python script on the current file
        python scripts/edit_dist.py "$FILE" "$OUTPUT"
        echo "Finished computing edit distance. Find the results in $OUTPUT"
    fi
done
