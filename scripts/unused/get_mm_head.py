import os

def get_mm_head(mm_file_path, output_file_path, num_lines=10):
    """
    Read the first `num_lines` lines from a Matrix Market (.mm) file.

    Parameters:
        file_path (str): Path to the .mm file
        num_lines (int): Number of lines to read from the start of the file
    
    Returns: 
        A list of the first `num_lines` lines from the file
    """
    with open(mm_file_path, 'r') as mm_file, open(output_file_path, 'w') as out_file:
        for _ in range(num_lines): # Iterate over the first num_line lines
            line = mm_file.readline()
            if not line:  # Break if the end of the file is reached
                break
            out_file.write(line) # Write the line to the output file

def get_mm_tail(mm_file_path, output_file_path, num_lines=10):
    """
    Read the last `num_lines` lines from a Matrix Market (.mm) file.

    Parameters:
        mm_file_path (str): Path to the .mm file.
        output_file_path (str): Path to the output file where the lines should be written.
        num_lines (int): Number of lines to read from the end of the file. Default is 10.
    """
    # Read the whole file
    with open(mm_file_path, 'r') as mm_file:
        lines = mm_file.readlines()

    # Get the last `num_lines` lines
    tail_lines = lines[-num_lines:]

    # Write the tail lines to the output file
    with open(output_file_path, 'w') as out_file:
        for line in tail_lines:
            out_file.write(line)

mm_file_path = os.path.join('data','wiki_bow.mm')
head_file = os.path.join('data','head_wiki_bow.txt')
tail_file = os.path.join('data','tail_wiki_bow.txt')
get_mm_head(mm_file_path, head_file, num_lines=100)
get_mm_tail(mm_file_path, tail_file, num_lines=100)