from __future__ import print_function
import os
import sys
import json
import numpy as np
from collections import Counter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import Dictionary


def create_question_dictionary(dataroot):
    dictionary = Dictionary()
    questions = []
    files = [
        'v2_OpenEnded_mscoco_train2014_questions.json',
        'v2_OpenEnded_mscoco_val2014_questions.json',
        'v2_OpenEnded_mscoco_test2015_questions.json',
        'v2_OpenEnded_mscoco_test-dev2015_questions.json'
    ]
    dictionary.add_word('<pad>')
    for path in files:
        question_path = os.path.join(dataroot, path)
        qs = json.load(open(question_path))['questions']
        for q in qs:
            dictionary.tokenize(q['question'], True)
    return dictionary


def create_caption_dictionary(dataroot, thres):
    dictionary = Dictionary()
    counter = Counter()
    files = [
        'captions_train2014.json',
        'captions_val2014.json',
    ]
    for path in files:
        caption_path = os.path.join(dataroot, path)
        qs = json.load(open(caption_path))['annotations']
        for q in qs:
            counter.update(dictionary.word_token(q['caption']))

    dictionary.add_word('<pad>')
    dictionary.add_word('<start>')
    dictionary.add_word('<end>')
    dictionary.add_word('<unk>')
    for word, cnt in counter.items():
        if cnt >= thres:
            dictionary.add_word(word)
    return dictionary


def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = map(float, vals[1:])
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb


if __name__ == '__main__':
    qd = create_question_dictionary('data')
    qd.dump_to_file('data/question_dictionary.pkl')
    word_cnt_thr = 5
    cd = create_caption_dictionary('data', word_cnt_thr)
    cd.dump_to_file('data/caption_dictionary.pkl')

    qd = Dictionary.load_from_file('data/question_dictionary.pkl')
    cd = Dictionary.load_from_file('data/caption_dictionary.pkl')
    emb_dim = 300
    glove_file = 'data/glove/glove.6B.%dd.txt' % emb_dim
    Qweights, Qword2emb = create_glove_embedding_init(qd.idx2word, glove_file)
    Cweights, Cword2emb = create_glove_embedding_init(cd.idx2word, glove_file)
    np.save('data/glove6b_init_%dd_question.npy' % emb_dim, Qweights)
    np.save('data/glove6b_init_%dd_caption.npy' % emb_dim, Cweights)
