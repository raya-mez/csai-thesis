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

mm_file_path = 'data/wiki_bow.mm'
output_file_path = 'data/head_wiki_bow.txt'
head_lines = get_mm_head(mm_file_path, output_file_path, 100)