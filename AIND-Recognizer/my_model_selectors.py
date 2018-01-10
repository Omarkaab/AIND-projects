import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        
        # Initializing
        best_BIC = float('inf')
        
        try:
        
            # Train the models for each number of states
            for i in range (self.min_n_components, self.max_n_components+1):
                model = self.base_model(i) # or GaussianHMM(n_components=i, n_iter=1000).fit(self.X, self.lengths) ?
                logL = model.score(self.X, self.lengths)
                # BIC formula
                BIC = -2 * logL + i * np.log(sum(self.lengths))
                # Do we have a better model i.e. a lower BIC?
                if BIC < best_BIC:
                    best_BIC = BIC
                    best_model = model
                
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, i))
            return None
        
        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Initializing
        best_DIC_score = float('-inf')
        best_model = None
        
        try:
        
            # Train the models for each number of states
            for i in range (self.min_n_components, self.max_n_components+1):
                
                # Train a model with i states            
                model = GaussianHMM(n_components=i, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(
                                                self.X, self.lengths)
                
                # Compute DIC score
                penalty = 0
                for other_word in self.words:
                    if other_word != self.this_word:
                        penalty += model.score(*self.hwords[other_word])
                        
                DIC_score = model.score(self.X, self.lengths) - 1/(len(self.hwords)-1) * penalty
                         
                # Do we have a better model i.e. a higher DIC score?
                if DIC_score > best_DIC_score:
                    best_DIC_score = DIC_score
                    best_model = model
                    
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, i))
            if best_model is None:
                return self.base_model(self.min_n_components)
            else:
                return best_model
            
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        # Initializing
        best_CV_score = float('-inf')
        best_model = None
        
        # Special case of not enough data points
        if len(self.sequences) < 3:
            return self.base_model(self.min_n_components)

        # KFold split
        split_method = KFold()
        split_gen = [[a,b] for a,b in split_method.split(self.sequences) ]
        
        # Train the models for each number of states
        for i in range (self.min_n_components, self.max_n_components+1):
            
            # Get next split
            train_set_idx, CV_set_idx = split_gen[i%3]

            # Train a model with i states            
            model = GaussianHMM(n_components=i, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(
                                            *combine_sequences(train_set_idx, self.sequences))
            
            try:
                # Score on the CV set
                CV_score = model.score(*combine_sequences(CV_set_idx, self.sequences))
            except:
                if best_model is None:
                    return self.base_model(self.min_n_components)
                else:
                    return best_model
                
            # Do we have a better model i.e. a higher CV score?
            if CV_score > best_CV_score:
                best_CV_score = CV_score
                best_model = model

        return best_model



