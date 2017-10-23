import numpy as np
import json
import codecs
import re
import itertools
from collections import Counter, OrderedDict

# 'coco' is a dict of image_id to sents used to make the dataset

def make_image_dict(coco):
    anns = coco['annotations']
    image_dict = {}
    for i in range(len(anns)):
        ann = anns[i]
        ann['caption'] = re.sub('[\n\r.]', '', ann['caption'])
        try:
            image_dict[ann['image_id']].append(ann['caption'])
        except:
            image_dict[ann['image_id']] = [ann['caption']]

    return image_dict

def get_sentences(im_dict):
    orig, altered = [], []
    count = 0
    for key in im_dict.keys():
        sents = im_dict[key]
        for i, sent in enumerate(sents):
            sents[i] = sent.strip().lower()
        orig.append(sents[0])
        altered.append(sents[1])
        orig.append(sents[2])
        altered.append(sents[3])
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


def process(coco):
    vocab_size = 10000
    im_dict = make_image_dict(coco)
    orig, altered = get_sentences(im_dict)
    # print(len(orig), len(altered))
    orig_tokenized = tokenize(orig)
    altered_tokenized = tokenize(altered)
    data_orig, data_altered, count, dictionary, reverse_dictionary = build_dataset(orig_tokenized, altered_tokenized, vocab_size)

    total_sents = len(orig)
    orig = np.array(orig)
    altered = np.array(altered)
    
    perm = np.random.permutation(total_sents)
    train_limit = int((0.97)*total_sents)
    valid_limit = int((0.98)*total_sents)
    orig_train = orig[perm[:train_limit]]
    orig_valid = orig[perm[train_limit:valid_limit]]
    orig_test = orig[perm[valid_limit:]]
    altered_train = altered[perm[:train_limit]]
    altered_valid = altered[perm[train_limit:valid_limit]]
    altered_test = altered[perm[valid_limit:]]

    # Create datafiles and vocab file
    orig_count = create_datafile(orig_train, "coco_tr.src")
    altered_count = create_datafile(altered_train, "coco_tr.tgt")
    orig_v_count = create_datafile(orig_valid, "coco_v.src")
    altered_v_count = create_datafile(altered_valid, "coco_v.tgt")
    orig_test_count = create_datafile(orig_test, "coco_test.src")
    altered_test_count = create_datafile(altered_test, "coco_test.tgt")
    
    
    vocab_count = create_vocab_file(
        reverse_dictionary, "coco_vocab" + str(vocab_size) + ".src")
    
    print("src count: %d %d %d tgt count: %d %d %d, vocab_count: %d" % (
        orig_count, orig_v_count, orig_test_count,
        altered_count, altered_v_count, altered_test_count, vocab_count))

    
    
desc = open("annotations/captions_train2017.json", "rb")
coco = json.loads(desc.readline())
process(coco)

