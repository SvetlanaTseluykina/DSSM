import collections
import itertools

import numpy as np
import re


def gen_trigrams():
    """
      Generates all trigrams for characters from trigram_chars
    """
    trigram_chars = "abcdefghijklmnopqrstuvwxyz"
    t3 = [''.join(x) for x in itertools.product(trigram_chars, repeat=3)]
    t2_start = ['#' + ''.join(x) for x in itertools.product(trigram_chars, repeat=2)]
    t2_end = [''.join(x) + '#' for x in itertools.product(trigram_chars, repeat=2)]
    t1 = ['#' + ''.join(x) + '#' for x in itertools.product(trigram_chars)]
    trigrams = t3 + t2_start + t2_end + t1
    vocab_size = len(trigrams)
    trigram_map = dict(zip(trigrams, range(1, vocab_size + 1)))  # trigram to index mapping, indices starting from 1
    return trigram_map


trigram_map = gen_trigrams()


def sentences_to_bag_of_trigrams(sentences):
    """
       Converts a sentence to bag-of-trigrams
       sentences: list of strings
       trigram_BOW: return value, (len(sentences),len(trigram_map)) size array
    """
    trigram_BOW = np.zeros((len(sentences), len(trigram_map) + 1), dtype='float32')  # one row for each sentence
    filter_pat = r'[\!"#&\(\)\*\+,-\./:;<=>\?\[\\\]\^_`\{\|\}~\t\n]'  # characters to filter out from the input
    for j, sent in enumerate(sentences):
        sent = re.sub(filter_pat, '', sent).lower()  # filter out special characters from input
        sent = re.sub(r"(\s)\s+", r"\1", sent)  # reduce multiple whitespaces to single whitespace
        words = sent.split(' ')
        indices = collections.defaultdict(int)
        for word in words:
            word = '#' + word + '#'
            for k in range(len(word) - 2):  # generate all trigrams for word word and update indices
                trig = word[k:k + 3]
                idx = trigram_map.get(trig, 0)
                indices[idx] = 1
        for key, val in indices.items():  # covert indices dict to np array
            trigram_BOW[j, key] = val
    return trigram_BOW
