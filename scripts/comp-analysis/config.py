import os
import pickle as pkl
from gensim.models import LsiModel

class Experiment():
    def __init__(self, ):
        self.cos_dist_types = ['raw', 'norm', 'abs', 'ang']
        self.form_dist_types = ['edit', 'edit_norm', 'jaccard']
        self.baselines = ['random', 'binned']
        self.baselines_abrev = ['rd', 'bin']
        self.word_lengths = [3,4,5,6,7]
        
        
        # Define paths
        # Input files
        # Vocabularies
        self.vocab_path = os.path.join('data', 'vocab.pkl')
        self.vocab_controlled_path = os.path.join('data', 'vocab_controlled.pkl')
        
        # Word ids keyed by word length
        self.ids_by_wordlength_path = os.path.join('data', 'ids_by_wordlength.pkl')
        self.ids_by_wordlength_controlled_path = os.path.join('data', 'ids_by_wordlength_controlled.pkl')
        
        # Models
        self.lsa_model_path = os.path.join('models', 'lsa', 'wiki_lsi_model.model')
        
        # Output files
        # Distance scores
        self.dist_scores_dir = os.path.join('results', 'distance_scores')
        self.dist_scores_file = os.path.join(self.dist_scores_dir, 'all_dist_scores.csv')
        self.avg_dist_file = os.path.join(self.dist_scores_dir, 'avg_dist.csv')
        
        # Correlation files (real, baselines, differences between real and baselines)
        self.corr_dir = os.path.join('results', 'correlation_scores')
        self.real_corr_scores_file = os.path.join(self.corr_dir, 'real_corr_scores.csv')
        self.rd_bl_corr_scores_file = os.path.join(self.corr_dir, 'rd_bl_corr_scores.csv')
        self.bin_bl_corr_scores_file = os.path.join(self.corr_dir, 'bin_bl_corr_scores.csv')
        self.corr_diffs_file = os.path.join(self.corr_dir, 'corr_diffs.csv')
        
        # z-score of real correlations compared to baseline
        self.zscores_file = os.path.join('results', 'zscores.csv')
    
    def load_vocabulary(self, vocab_path):
        """
        Load the pickled vocabulary from the specified file path and returns it. 
        The vocabulary is a dictionary containing words as values and their corresponding word IDs as keys.
        """
        with open(vocab_path, 'rb') as f:
            vocabulary = pkl.load(f)
        return vocabulary
    
    def word_ids_by_word_length(self, vocab):
        """Groups word IDs in the vocabulary by the length of their correponsing words. 

        Args:
            vocab (dict): Dictionary containing words keyed by their IDs.

        Returns:
            dict: List of word IDs keyed by word length
        """        
        words_ids_by_length = {}
        for length in range(3,8):
            ids = [id for id, word in vocab.items() if len(word) == length] 
            words_ids_by_length[length] = ids
        return words_ids_by_length
    
    def get_vocabulary_embeddings_dict(self, vocab): 
        """
        Extract the embeddings of the words in the vocabulary.
        
        The function loads a vocabulary dictionary and the LSA model. 
        It then retrieves the embeddings of the words in the vocabulary
        from the `projection.u` attribute of the LSI model.
        
        Args:
            lsa_model (str): The path to the LSI model file. Defaults to "models/wiki_lsi_model.model".
            vocab (dict): The dictionary where keys are word IDs and values are the corresponding words.
            
        Returns:
            dict: A dictionary mapping word IDs in the vocabulary to their embeddings.
        """
        
        # Load the LSI model from the specified path
        lsa_model = LsiModel.load(self.lsa_model_path)    
        
        # Create a dictionary mapping words to their embeddings (the left singular vectors from the LSI model)
        embeddings_dict = {id:lsa_model.projection.u[id] for id in vocab.keys()}
        
        # Return the dictionary of words and their embeddings
        return embeddings_dict



class Test():
    def __init__(self):
        self.word_list = ['banana', 'bandana', 'league', 'cup', 'cap']

class Globals:
    def __init__(self):
        # Define internal variables
        self._word_lengths = [3, 4, 5, 6, 7]
        self._cos_dist_types = ['raw', 'norm', 'abs', 'ang']
        self._form_dist_types = ['edit', 'edit_norm', 'jaccard']
        
        self._rescaling_types_abrev = ['none', 'abs', 'norm', 'ang']
        self._rescaling_types_args = [None, 'abs_cos_sim', 'norm_cos_sim', 'ang_dist']
        self._rescaling_types_dict = {
            'none': None,
            'abs': 'abs_cos_sim',
            'norm': 'norm_cos_sim',
            'ang': 'angular_dist'}
        self._baseline_types_abrev = ['rd', 'bin']
        self._baseline_types = ['Random Baseline', 'Binned Baseline']

    # Define properties to make attributes read-only
    @property
    def word_lengths(self):
        return self._word_lengths

    @property
    def cos_dist_types(self):
        return self._cos_dist_types
    
    @property
    def form_dist_types(self):
        return self._form_dist_types
    
    @property
    def rescaling_types_abrev(self):
        return self._rescaling_types_abrev

    @property
    def rescaling_types_args(self):
        return self._rescaling_types_args

    @property
    def rescaling_types_dict(self):
        return self._rescaling_types_dict

    @property
    def baseline_types_abrev(self):
        return self._baseline_types_abrev

    @property
    def baseline_types(self):
        return self._baseline_types
    
    