from collections import defaultdict 

files_fields_dict = {'../Data/Celex/Celex/english/esl/esl.cd':{'idnum':0, 'head':1, 'classnum':3},
                    '../Data/Celex/Celex/english/eml/eml.cd':{'idnum':0, 'morphstatus':3},
                    '../Data/Celex/Celex/english/eol/eol.cd':{'idnum':0, 'syllables':7}
                    }

def create_initial_wfdict(filepath='../Data/Celex/Celex/english/eow/eow.cd'):
    """Creates a dictionary keyed by wordform id number as instantiated in CELEX, 
    and with values - the corresponding wordform with diacritics and the id number of the word lemma."""
    wfdict = defaultdict(dict)
    with open(filepath, 'r') as file:
        for line in file:
            fields = line.strip().split("\\")
            wf_idnum, wf_worddia, wf_idnumlemma = fields[0], fields[1], fields[3]
            wfdict[wf_idnum].update({"worddia": wf_worddia, "idnumlemma": wf_idnumlemma})
    return wfdict

def create_infodict(files_fields_dict=files_fields_dict):
    """
    Creates a dictionary word information from specified files and fields within them.
    
    Parameters:
    - files_fields_dict: A dictionary where each key is a file path and each value is another dictionary.
        The value dictionaries map field names to their indices in the respective file.
    
    Returns:
    - A defaultdict containing the aggregated lemma-related data from the specified fields of the given files.
    """      
    
    num_to_class = {  
        "1"  : "N",      # Noun
        "2"  : "A",      # Adjective
        "3"  : "NUM",    # Numeral
        "4"  : "V",      # Verb
        "5"  : "ART",    # Article
        "6"  : "PRON",   # Pronoun
        "7"  : "ADV",    # Adverb
        "8"  : "PREP",   # Preposition
        "9"  : "C",      # Complementizer
        "10" : "I",      # Interjection
        "11" : "SCON",   # Subordinating conjunction
        "12" : "CCON",   # Coordinating conjunction
        "13" : "LET",    # Letter
        "14" : "ABB",    # Abbreviation
        "15" : "TO"      # Infinitive marker "to"
        }
    
    # Initialize the dictionary
    infodict = defaultdict(dict)
    
    # Iterate over the files and extract the information from the relevant fields
    for filepath, fields in files_fields_dict.items():
        with open(filepath, 'r') as file:
            for line in file:
                fields_data = line.strip().split("\\")
                # All lemmas are universally idenfied by a unique id number
                idnum = fields_data[fields['idnum']] 
                
                for field_name, field_index in fields.items():
                    # Add the information from the fields to infodict
                    # Replace the class number with the corresponding grammatical class abbreviation
                    if field_name == 'classnum':
                        infodict[idnum]['class'] = num_to_class[str(fields_data[field_index])]
                    # Add the information from the remaining fields (making sure the field index is not out of range)
                    if field_name != 'idnum' and field_index < len(fields_data):
                        # Populate infodict with meaningful field names and values
                        infodict[idnum][field_name] = fields_data[field_index]

    return infodict

def enrich_wfdict(wfdict, infodict):
    """
    Enriches wfdict entries with information from infodict based on lemma ID numbers.
    
    Parameters:
    - wfdict: Dictionary containing word form information.
    - infodict: Dictionary containing detailed information about lemmas.
    
    Updates wfdict in-place by adding information from infodict.
    """    
    
    for _, wf_info in wfdict.items():
        idnumlemma=wf_info["idnumlemma"]
        wf_info["lemma"]=infodict[idnumlemma]["head"]
        wf_info["class"]=infodict[idnumlemma]["class"]
        wf_info["morphstatus"]=infodict[idnumlemma]["morphstatus"]
        wf_info["syllables"]=infodict[idnumlemma]["syllables"]
    return wfdict

def get_wfdict(files_fields_dict):
    """
    Compiles a dictionary with wordforms and information about them 
    in a single dictionary based on the CELEX database.
    
    Parameters:
    - files_fields_dict: A dictionary specifying files and fields to extract lemma-related information.
    
    Returns:
    - wfdict: A combined dictionary with wordforms and related linguistic information.
    """
    # Create wfdict with basic word form information (wordform with diacritics and lemma id)
    wfdict = create_initial_wfdict()

    # Create infodict with detailed linguistic information specified in the argument
    infodict = create_infodict(files_fields_dict)
    
    # Combine wfdict and infodict
    enrich_wfdict(wfdict, infodict)
    
    # Returned the enriched wfdict with the information from infodict
    return wfdict