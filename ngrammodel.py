# ngram language model code
# majority of code taken from and modified from the NLP Lab1 exercise

from collections import Counter
from math import log

def glue_tokens(tokens, order):
    """A useful way of glueing tokens together for
    Kneser Ney smoothing and other smoothing methods
    
    :param: order is the order of the language model
        (1 = unigram, 2 = bigram, 3 =trigram etc.)
    """
    return '{0}@{1}'.format(order,' '.join(tokens))

def unglue_tokens(tokenstring, order):
    """Ungluing tokens glued by the glue_tokens method"""
    if order == 1:
        return [tokenstring.split("@")[1].replace(" ","")]
    return tokenstring.split("@")[1].split(" ")

def add_s_tags(tokens, order):
    """Returns a list of tokens with the correct numbers of initial
    and end tags (this is meant to be used with a non-backoff model!!!)
    
    :tokens: list of tokens from a sentence
    :param: order is the order of the language model
        (1 = unigram, 2 = bigram, 3 =trigram etc.)
    """
    tokens = ['<s>'] * (order-1) + tokens + ['</s>']
    return tokens

# returns a hashmap with key=unigrams, value=frequency in corpus
# type=1: gender, match="male"
def countUnigrams(sentences): 
    unigrams = Counter()
    for sent in sentences:
        tokens = add_s_tags(sent, 1) #comment
        for w in tokens:
            unigrams[w] +=1
    return unigrams

# takes a hashmap of unigrams and replaces anything occuring less than the frequency 
# with out-of-vocab symbol <unk/>
# returns a new list: vocab
def minDocFrequency(unigrams, frequency):
    vocab = Counter()
    for x in unigrams:
        if unigrams[x] < frequency: 
            vocab['<unk/>'] += 1 
        else:
            vocab[x] = unigrams[x]
    return vocab

# takes a list of words and checks if each word is in our vocab, if it's not we replace with <unk/>
# returns a new list of words containing only words in our vocab
def replaceIfNotInVocab(words, vocab):
    replacedWords = []
    for w in words:
        if w not in vocab:
            if w != "<s>": # we don't want to add <s> to our vocab for bigrams upwards since it's never a continuation word
                replacedWords.append("<unk/>")
            else:
                replacedWords.append(w)
        else:
            replacedWords.append(w)
    return replacedWords

# Kneser-Ney smoothing
def ngrams_interpolated_kneser_ney(tokens,
                                   order,
                                   ngram_numerator_map,
                                   ngram_denominator_map,
                                   ngram_non_zero_map,
                                   unigram_denominator):
    """Function used in n-gram language model training
    to count the n-grams in tokens and also record the
    lower order non -ero counts necessary for interpolated Kneser-Ney
    smoothing.
    
    Taken from Goodman 2001 and generalized to arbitrary orders"""
    for i in range(order-1,len(tokens)): # tokens should have a prefix of order - 1
        #print i
        for d in range(order,0,-1): #go through all the different 'n's
            if d == 1:
                unigram_denominator += 1
                ngram_numerator_map[glue_tokens(tokens[i],d)] += 1
            else:
                den_key = glue_tokens(tokens[i-(d-1) : i], d)
                num_key = glue_tokens(tokens[i-(d-1) : i+1], d)
    
                ngram_denominator_map[den_key] += 1
                # we store this value to check if it's 0
                tmp = ngram_numerator_map[num_key]
                ngram_numerator_map[num_key] += 1 # we increment it
                if tmp == 0: # if this is the first time we see this ngram
                    #number of types it's been used as a context for
                    ngram_non_zero_map[den_key] += 1
                else:
                    break 
                    # if the ngram has already been seen
                    # we don't go down to lower order models
    return ngram_numerator_map, ngram_denominator_map, ngram_non_zero_map, unigram_denominator

def kneser_ney_ngram_prob(ngram, discount, order, ngram_numerator_map, ngram_denominator_map, ngram_non_zero_map, unigram_denominator):
    """KN smoothed ngram probability from Goodman 2001.
    This is run at test time to calculate the probability
    of a given n-gram or a given order with a given discount.
    
    ngram :: list of strings, the ngram
    discount :: float, the discount used (lambda)
    order :: int, order of the model
    """
    # First, calculate the unigram prob of the last token 
    # If we've never seen it at all, it will 
    # have no probability as a numerator
    uni_num = ngram_numerator_map.get(glue_tokens(ngram[-1], 1))
    if not uni_num: # if no value found in dict, make it 0
        uni_num = 0
    probability = previous_prob = float(uni_num) / float(unigram_denominator)
    
    # Given <unk/> should have been used in place of unknown words before passing
    # to this method,
    # probability should be non-zero
    if probability == 0.0:
        print("0 prob for unigram!")
        print(glue_tokens(ngram[-1], 1))
        print(ngram)
        print(ngram_numerator_map.get(glue_tokens(ngram[-1], 1)))
        print(unigram_denominator)
        raise Exception

    # Compute the higher order probs (from 2/bi-gram upwards) and interpolate them
    for d in range(2,order+1):
        # Get the number of times this denominator has been seen as one
        # For bigrams this is the number of different continuation types counted
        ngram_den = ngram_denominator_map.get(glue_tokens(ngram[-(d):-1], d))
        if not ngram_den: # if no value found in dict, make it 0
            ngram_den = 0
        if ngram_den != 0: 
            ngram_num = ngram_numerator_map.get(glue_tokens(ngram[-(d):], d))
            if not ngram_num: # if no value found in dict, make it 0
                ngram_num = 0
            if ngram_num != 0:
                current_prob = (ngram_num - discount) / float(ngram_den)
            else:
                current_prob = 0.0
            nonzero = ngram_non_zero_map.get(glue_tokens(ngram[-(d):-1], d))
            if not nonzero: # if no value found in dict, make it 0
                nonzero = 0
            # interpolate with previous probability of lower orders calculated
            # so far
            current_prob += nonzero * discount / ngram_den * previous_prob
            previous_prob = current_prob
            probability = current_prob
        else:
            #if this context (e.g. bigram contect for trigrams) has never been seen, 
            #then we can only get the last order with a probability (e.g. unigram)
            #and halt
            probability = previous_prob
            break
    return probability

# function to calculate kneser_ney_ngram_prob
# returns perplexity
def perplexityKN(sentences, vocab, order, discount, boolPrint, trainedKN):
    ngram_numerator_map = trainedKN[0]
    ngram_denominator_map = trainedKN[1]
    ngram_non_zero_map = trainedKN[2]
    unigram_denominator = trainedKN[3]
    s = 0  # total neg log prob mass for cross entropy
    N = 0 # total number of words for normalizing s
    delta = order - 1
    for words in sentences:
        words = add_s_tags(words, order)
        words = replaceIfNotInVocab(words, vocab)
        sent_s = 0  # recording non-normalized entropy for this sentence
        sent_N = 0  # total number of words in this sentence (for normalization)
        for i in range(delta, len(words)):
            context = words[i-delta:i]
            target = words[i]
            ngram = context + [target]
            prob = kneser_ney_ngram_prob(ngram, discount, order, ngram_numerator_map, ngram_denominator_map, ngram_non_zero_map, unigram_denominator)
            s += -log(prob, 2) # add the neg log prob to s
            sent_s += -log(prob, 2)  # add the neg log prob to sent_s
            N += 1 # increment the number of total words
            sent_N += 1 # increment the number of total words in this sentence
    sent_cross_entropy = sent_s/sent_N
    sent_perplexity = 2 ** sent_cross_entropy
    # print(words, "cross entropy:", sent_cross_entropy, "perplexity:", sent_perplexity)
    cross_entropy = s/N
    perplexity = 2 ** cross_entropy
    if boolPrint == True:
        # print(discount)
        #print("corpus cross entropy", cross_entropy)
        #print("corpus perplexity", perplexity)
        print("{:.2f}".format(perplexity))
    return perplexity

# a function to train the model
def trainKN(trainData, vocab, order):
    # reset the below each time we run training
    unigram_denominator = 0
    ngram_numerator_map = Counter() 
    ngram_denominator_map = Counter() 
    ngram_non_zero_map = Counter()
    
    for tokens in trainData:
        new_tokens = add_s_tags(tokens, order)
        new_tokens = replaceIfNotInVocab(new_tokens, vocab)
        # print(new_tokens)
        ngram_numerator_map, ngram_denominator_map, ngram_non_zero_map, unigram_denominator =\
                ngrams_interpolated_kneser_ney(new_tokens,
                                               order,
                                               ngram_numerator_map,
                                               ngram_denominator_map,
                                               ngram_non_zero_map,
                                               unigram_denominator)
    return (ngram_numerator_map, ngram_denominator_map, ngram_non_zero_map, unigram_denominator)

