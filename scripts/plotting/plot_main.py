import logging
import sys
import scripts.plotting.OLD_plotting_funcs as OLD_plotting_funcs

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s : %(levelname)s : %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

if len(sys.argv) < 3:
    sys.exit(1)

# File paths
input_csv_file =sys.argv[1] # "results/avg_dist.csv"
output_image_file = sys.argv[2] # "plots/avg_dist_scatterplot.png"

