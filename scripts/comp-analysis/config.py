class Experiment():
    def __init__(self, cos_dist_type, form_dist_type, baselines, word_lengths=[3,4,5,6,7]):
        self.cos_dist_type = cos_dist_type
        self.form_dist_type = form_dist_type
        self.baselines = baselines
        self.word_lengths = word_lengths
        # TODO: add paths

class Test():
    def __init__(self):
        self.word_list = ['banana', 'bandana', 'league', 'cup', 'cap']

class Globals:
    def __init__(self):
        # Define internal variables
        self._word_lengths = [3, 4, 5, 6, 7]
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
    
    