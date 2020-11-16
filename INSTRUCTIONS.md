# 1. General running instructions
The IPython notebook is pre-setup with the best features, classifier and classifier parameters that were found. It should be able to be run all the way through, as is, to result in close to the optimal performance obtained.
# 2. Adding and removing features
There are several places where key parameters can be amended to change the features/classifiers. All parameters can generally be enabled by a simple flag with no other changes required.
The main place to change the features included is in the parameters of the preProcess function. The parameters are currently set as (punc=1, stem=1, stop=2, lemma=0, bigrams=0, crftag=3) and these are explained in greater detail in code comments.

There are four global boolean variables at the top of the notebook.ngramGender and ngramCharacter control whether to add the perplexity from n-gram language models as features. Several large functions for the KN smoothing relating to this have been hidden in the separate ngrammodel.py file in the project directory, and then imported into the notebook. Calls to this file from the IPython notebook are prefaced with ngram.function() syntax.

The two other global variables at the top are flags to add the stopsRemoved and sentenceLength features.
# 3. Changing classifier type
Different classifiers can be enabled by commenting lines in the trainClassier() function.
