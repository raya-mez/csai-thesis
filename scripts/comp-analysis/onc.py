import re
import numpy as np
import pickle

# definition of nuclei in written English
class en():
    def __init__(self):
        self.double_letters = ['aa', 'ea', 'ee', 'ia', 'ie',
                                'io(?!u)', 'oo', 'oe', 'ou', '(?<!q)ui(?=.)', 'ei', 'eu', 'ae', 'ey(?=.)', 'oa']
        self.single_letters = ['a', 'e', 'i(?!ou)', 'o', '(?<!q)u', 'y(?![aeiou])']
        self.accented_letters = [u'à', u'ê', u'è', u'é', u'â', u'ô', u'ü']
        self.double_letter_pattern = '|'.join(self.double_letters)
        self.single_letter_pattern = '|'.join(self.single_letters)
        self.accented_letter_pattern = '|'.join(self.accented_letters)
        self.nucleuspattern = '%s|%s|%s' % (
        self.double_letter_pattern, self.accented_letter_pattern, self.single_letter_pattern)
        self.oncpattern = re.compile('(.*?)(%s)(.*)' % self.nucleuspattern)

def onc_syllable(syllable, lang=None, output='list'):
    """
    Segments a given syllable into onset, nucleus, and coda components (given the pattern for the given language).
    
    Parameters:
    - syllable (str): The syllable to be segmented.
    - lang (object): An object representing a specific language, which must have an `oncpattern` attribute 
        (a compiled regular expression) for syllable segmentation.
    - output (str): The format of the function's return value. Possible values: 'list' or 'string. 
    
    Returns:
    - list or str: A list or a string of the segmented syllable components (onset, nucleus, coda), 
        depending on the `output` parameter. 
        If 'list' (default), returns a list of the segmented components.
        If 'string', returns a string with the components joined by colons. 
    
    Raises:
    - AttributeError: If the syllable cannot be segmented using the provided language's pattern.
    - ValueError: If the `output` parameter is neither 'list' nor 'string'.
    """
    
    # Retrieve the regular expression pattern for syllable segmentation from the language object
    oncpattern = lang.oncpattern
    
    # Attempt to match the syllable against the segmentation pattern
    m = oncpattern.match(syllable)
    
    try:
        # If a match is found, extract the onset, nucleus, and coda components from the match groups
        sequence = [m.group(1), m.group(2), m.group(3)]
    except AttributeError as err:
        # Raise an AttributeError if the syllable could not be segmented (e.g., no match found)
        raise AttributeError('Input syllable could not be segmented') from err
    
    # Return the segmented components based on the specified output format
    if output == 'list':
        return sequence
    elif output == 'string':
        # Join the components with colons if 'string' output is specified
        return ':'.join(sequence)
    else:
        # Raise a ValueError if an unsupported output format is specified
        raise ValueError('output must be `list` or `string')

def onc_word(word, lang=None, output='list'):
    """
    Segments each syllable of the given word into onset, nucleus, 
    and coda components based on a pattern provided language's pattern.
    
    Parameters:
    - word (str): The word to be segmented into its syllable components, 
        its syllables being separated by hyphens (e.g., 'he-li-cop-ter').
    - lang (object): An object representing a specific language, which must have an `oncpattern` attribute 
        (a compiled regular expression) determining the syllable segmentation pattern.
    - output (str): The format of the function's return value. Possible values: 'list' (default) or 'string'.
        If 'list', returns a list where each element is the segmented components of a syllable. 
        If 'string', returns a single string with syllables and their components properly joined.
    
    Returns:
    - list or str: Depending on the `output` parameter, either a list of lists (each representing the segmented 
        components of a syllable) or a string with syllables and their components joined by hyphens and colons, 
        respectively. 

    Raises:
    - ValueError: If the `output` parameter is neither 'list' nor 'string'.
    """
    
    # Segment each syllable in the word into onset, nucleus, and coda components
    # The word is split into syllables by hyphens, and each syllable is processed individually
    sequence = [onc_syllable(syllable, lang, output=output) for syllable in word.split('-')]
    
    # Return the segmented components based on the specified output format
    if output == 'list':
        # If 'list' is specified, return the sequence of segmented syllables as it is
        return sequence
    elif output == 'string':
        # If 'string' is specified, join the segmented syllable components with hyphens
        return '-'.join(sequence)
    else:
        # Raise a ValueError if an unsupported output format is specified
        raise ValueError('output must be `list` or `string`')

# Initialize a global variable _wfdict to None that will hold the loaded dictionary
# to ensure it's loaded only once during the module's lifetime.
_wfdict = None

# Lazy load the dictionary storing word information
def load_word_dict(filepath='../Data/wfdict.pkl'):
    """
    Lazily loads the word information dictionary from a pickle file and stores it in a global variable.
    Subsequent calls to this function will return the already loaded dictionary, avoiding redundant I/O operations.
    
    Parameters:
        filepath (str): Path to the pickle file containing the word dictionary. Defaults to '../Data/wfdict.pkl'.
    
    Returns:
        dict: The loaded word information dictionary.
    """
    # Declare _wfdict as global to modify the global instance within this function
    global _wfdict  
    # Check if the dictionary has not been loaded yet
    if _wfdict is None:  
        # Load the dictionary from the pickle file and assign it to _wfdict
        with open(filepath, 'rb') as f:  
            _wfdict = pickle.load(f)  
    # Return the loaded dictionary
    return _wfdict  

def syllabify(word):
    wfdict = load_word_dict()
    """
    Segments the input word into syllables using the word information dictionary.

    Parameters:
        word (str): The word to be syllabified.
        wfdict (dict): The dictionary containing word information, including syllabification.
    
    Returns:
        str: The syllabified word, with hyphens separating the syllables.

    Raises:
        ValueError: If the word is not found in the dictionary.
    """
    for entry in wfdict.values():
        if word in (entry.get('worddia'), entry.get('lemma')):
            return entry['syllables']
    
    raise ValueError(f"'{word}' is an unknown word")

# Define an ONC-based form similarity measure
def onc_similarity(word1, word2):    
    """
    Calculate the similarity between two words based on their Onset-Nucleus-Coda (ONC) composition.

    After syllabifiying each word, it extracts the ONC components of each syllable, and then compares these components
    to determine a similarity ratio. The similarity ratio is calculated as the number of matching ONC components
    (considering their position within the word) divided by the total number of ONC components considered across both words.
    To ensure a fair comparison, the shorter word is 'patched' by repeating its syllables until it matches the
    length of the longer word. 

    Parameters:
    - word1 (str): The first word to compare.
    - word2 (str): The second word to compare.

    Returns:
    - float: A ratio representing the similarity between the two words, ranging from 0 (no similarity)
        to 1 (identical ONC composition).
    """
    
    # Syllabify the words
    w1_syl = syllabify(word1)
    w2_syl = syllabify(word2)
    
    # Extract ONC from the syllabified words
    en_nucleus = en()
    w1_onc = onc_word(w1_syl, en_nucleus, output="list")
    w2_onc = onc_word(w2_syl, en_nucleus, output="list")

    # Patch the shorter word to match the length of the longer one
    # Find which word is longer and by how many syllables
    longer_word = w1_onc if len(w1_onc) >= len(w2_onc) else w2_onc
    shorter_word = w1_onc if len(w1_onc) < len(w2_onc) else w2_onc
    length_diff = abs(len(w1_onc) - len(w2_onc))
    # Initialize an empty list for the patched shorter word
    shorter_word_patched = shorter_word.copy()
    # Loop to append syllables until the shorter_word_patched matches the length of longer_word
    for _ in range(length_diff):
        # Find the next syllable to append using modulo to cycle through shorter_word syllables (if necessary)
        syllable_to_append = shorter_word[len(shorter_word_patched) % len(shorter_word)]
        shorter_word_patched.append(syllable_to_append)

    # Compare the ONC composition to get an overlap ratio
    match_array = np.zeros((len(longer_word), 3))

    for i, (syl1, syl2) in enumerate(zip(w1_onc, w2_onc)):
        match_array[i] = [
                1 if syl1[0] == syl2[0] else 0,  # Onset
                1 if syl1[1] == syl2[1] else 0,  # Nucleus
                1 if syl1[2] == syl2[2] else 0   # Coda
                ]   

    # Calculate the ratio of matches to total components
    total_matches = np.sum(match_array)
    total_components = np.prod(match_array.shape)
    match_ratio = total_matches / total_components
    
    return match_ratio