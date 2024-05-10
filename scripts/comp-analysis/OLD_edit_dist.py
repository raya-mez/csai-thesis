import logging
import sys
import os
import csv
import editdistance

# Configure logging to display information to the console
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

if len(sys.argv) < 3:
    print("Usage: python edit_dist.py <input_file> <output_file>")
    sys.exit(1)

cos_dist_scores_dir = sys.argv[1] # "results/cos_dist_per_word_length"
output_dir = sys.argv[2] # "results/edit_cos_dist"

for input_file in os.listdir(cos_dist_scores_dir):
    logging.info(f"Computing edit distance for file {input_file}")
    
    input_file_path = os.path.join(cos_dist_scores_dir, input_file)
    
    output_file = f"{output_dir}/edit_dist_{input_file}"
    
    # Open input and output CSV files
    with open(input_file_path, 'r', newline='') as input_csv, open(output_file, 'w', newline='') as output_csv:
        csv_reader = csv.reader(input_csv)
        csv_writer = csv.writer(output_csv)
        
        # Write a header line to the output CSV file
        csv_writer.writerow(['word1', 'word2', 'cos_dist', 'edit_dist'])
        
        # Process each row in the input CSV file
        for row in csv_reader:
            word1, word2, similarity_score = row
            # Calculate edit distance
            distance = editdistance.eval(word1, word2)
            # Append edit distance to the row
            row.append(distance)
            # Write the modified row to the output CSV file
            csv_writer.writerow(row)