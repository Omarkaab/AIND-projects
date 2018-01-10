import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    
    # Lopp on the test items
    for test_index in range(test_set.num_items):
        # Initializing the loop
        current_dict = {}
        best_guess_score = float('-inf')
        best_guess = None
        
        # apply the current test item to the trained models
        for model_word in models.keys():
            # Store the logL value in the dict
            model = models[model_word]
            try:
                current_dict[model_word] = model.score(*test_set.get_item_Xlengths(test_index))
                # Do we have better guess to keep in guesses?
                if current_dict[model_word] > best_guess_score:
                    best_guess_score = current_dict[model_word]
                    best_guess = model_word                
            except:
                # Skip this one
                current_dict[model_word] = float('-inf')
                pass  
              
        # Store what we have so far
        guesses.append(best_guess)
        probabilities.append(current_dict)


    return probabilities, guesses
