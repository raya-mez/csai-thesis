import bz2
import sys

def write_head(input_file_path, output_file_path, n_lines=10):
    # Open the compressed file for reading
    with bz2.open(input_file_path, 'rt') as file, open(output_file_path, 'w') as new_file:
        # Iterate over the first n_lines of the compressed file
        for _ in range(n_lines):
            line = file.readline()
            # Break the loop if the line is empty (end of file)
            if not line:
                break
            # Write the line to the new file
            new_file.write(line)

if __name__ == "__main__":
    # Check if enough arguments are passed
    if len(sys.argv) < 3:
        print("Usage: python write_head.py <input_file_path> <output_file_path> [n_lines]")
        sys.exit(1)
    
    # Paths are provided via command line arguments
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Number of lines is optional; default is 10
    n_lines = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    # Call the function with the command line arguments
    write_head(input_path, output_path, n_lines)