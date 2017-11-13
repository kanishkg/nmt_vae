import numpy as np
import json
import codecs
import re
import itertools
from collections import Counter, OrderedDict

# 'coco' is a dict of image_id to sents used to make the dataset

def get_sentences(quora):
    orig, altered = [], []
    count = 0
    for line in quora.readlines():
        segments = line.split("\t")
        if len(segments) == 6 and segments[-1].strip() == str(1):
            orig.append(segments[3].strip())
            altered.append(segments[4].strip())
    return orig, altered
    
def tokenize(sents):
    tokenized = []
    for sent in sents:
        tokenized.append(sent.split(" "))
    return tokenized

# build a vocabulary of words

def build_dataset(orig, altered, vocabulary_size):
    # print len(orig), len(altered)
    count = [['UNK', -1]]
    sentences = []
    for sentence in orig:
	sentences.append(sentence)
    for sentence in altered:
	sentences.append(sentence)
    
    count.extend(Counter(itertools.chain(*sentences)).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
	dictionary[word] = len(dictionary)
    data_orig = list()
    data_altered = list()
    unk_count = 0
    for ind, sentence in enumerate(orig):
        data_orig.append([])
	for word in sentence:
	    if word in dictionary:
		index = dictionary[word]
	    else:
		index = 0  # dictionary['UNK']
                unk_count = unk_count + 1
	    data_orig[ind].append(index)
    for ind, sentence in enumerate(altered):
	data_altered.append([])
	for word in sentence:
	    if word in dictionary:
		index = dictionary[word]
	    else:
		index = 0  # dictionary['UNK']
		unk_count = unk_count + 1
	    data_altered[ind].append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data_orig, data_altered, count, dictionary, reverse_dictionary

def create_datafile(sents, filename):
    ''' Create a file with each element of sents on a separate line.
    Args:
    @sents: list of sents to be put in file
    @filename: name of the file created

    returns: Number of sentences processed
    '''
    new_file = codecs.open(filename, "wb", "utf-8")
    for sent in sents:
        new_file.write(sent + "\n")
    return len(sents)

def create_vocab_file(rev_dict, filename):
    ''' Create a file with each word in the vocab on a separate line.'''
    new_file = codecs.open(filename, "wb", "utf-8")
    word_count = 0
    # The below line was throwing up error with unk's vocab code for some reason.
    # new_file.write("<unk>\n<s>\n</s>\n")
    for i, ind in enumerate(rev_dict.keys()):
        if i > 0 and rev_dict[i] != '':
            new_file.write(rev_dict[ind] + "\n")
            word_count += 1
    return word_count


def process(quora):
    vocab_size = 10000
    orig, altered = get_sentences(quora)
    # print(len(orig), len(altered))
    orig_tokenized = tokenize(orig)
    altered_tokenized = tokenize(altered)
    data_orig, data_altered, count, dictionary, reverse_dictionary = build_dataset(orig_tokenized, altered_tokenized, vocab_size)

    total_sents = len(orig)
    orig = np.array(orig)
    altered = np.array(altered)
    
    perm = np.random.permutation(total_sents)
    train_limit = int((0.90)*total_sents)
    valid_limit = int((0.97)*total_sents)
    orig_train = orig[perm[:train_limit]]
    orig_valid = orig[perm[train_limit:valid_limit]]
    orig_test = orig[perm[valid_limit:]]
    altered_train = altered[perm[:train_limit]]
    altered_valid = altered[perm[train_limit:valid_limit]]
    altered_test = altered[perm[valid_limit:]]

    # Create datafiles and vocab file
    orig_count = create_datafile(orig_train, "quora_tr.src")
    altered_count = create_datafile(altered_train, "quora_tr.tgt")
    orig_v_count = create_datafile(orig_valid, "quora_v.src")
    altered_v_count = create_datafile(altered_valid, "quora_v.tgt")
    orig_test_count = create_datafile(orig_test, "quora_test.src")
    altered_test_count = create_datafile(altered_test, "quora_test.tgt")
    
    
    vocab_count = create_vocab_file(
        reverse_dictionary, "quora_vocab" + str(vocab_size) + ".src")
    
    print("src count: %d %d %d tgt count: %d %d %d, vocab_count: %d" % (
        orig_count, orig_v_count, orig_test_count,
        altered_count, altered_v_count, altered_test_count, vocab_count))

    
    
desc = codecs.open("./quora_duplicate_questions.tsv", "rb", "utf-8")
process(desc)

