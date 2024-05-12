import os
import sys
import logging

def setup_logging():
    """Configure logging to display information to the console"""
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s : %(levelname)s : %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

def validate_paths(paths):
    """Validate if the provided paths are valid and accessible."""
    for path in paths:
        if not os.path.exists(path):
            logging.error("Path %s does not exist.", path)
            return False
        if not os.path.isfile(path):
            logging.error("Path %s is not a file.", path)
            return False
        if not os.access(path, os.R_OK):
            logging.error("No read access to %s.", path)
            return False
    return True

