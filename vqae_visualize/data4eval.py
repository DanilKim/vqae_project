from __future__ import print_function
import os
import json
import pickle
import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return self.word2idx['<pad>']

    def word_token(self, sentence):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        return words

    def tokenize(self, sentence, add_word):
        words = self.word_token(sentence)
        if add_word:
            tokens = [self.add_word(w) for w in words]
        else:
            tokens = [self.word2idx[w] if w in self.word2idx
                      else self.word2idx['<unk>'] for w in words]
        return tokens

    def dump_to_file(self, path):
        pickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = pickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer, caption):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer,
        'caption'     : caption}
    return entry


def _load_dataset(dataroot, name, img_id2val, thres):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    question_path = os.path.join(
        dataroot, 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])

    caption_path = os.path.join(
        dataroot, 'Cap_Att/Caption_Attention_results_%s.pickle' % name)
    captions = sorted(pickle.load(open(caption_path)),
                      key=lambda x: x['question_id'])

    from nltk.tokenize import word_tokenize
    qid2gtcap = {}
    for i, ex in enumerate(captions):
        m1 = list(ex['caption_attention_1'].values())
        m2 = list(ex['caption_attention_2'].values())
        if max(m1) < thres and max(m2) < thres:
            continue
        if max(m1) > max(m2):
            ps_gt_caption = list(ex['caption_attention_1'].keys())[m1.index(max(m1))]
        else:
            ps_gt_caption = list(ex['caption_attention_2'].keys())[m2.index(max(m2))]
        qid2gtcap[ex['question_id']] = ps_gt_caption

    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = pickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])

    utils.assert_eq(len(questions), len(answers))
    entries = []
    for question, answer in zip(questions, answers):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        if question['question_id'] in qid2gtcap:
            entries.append(_create_entry(img_id2val[img_id], question, answer,
                                         qid2gtcap[question['question_id']]))

    return entries


class VQAFeatureDataset(Dataset):
    def __init__(self, name, question_dict, caption_dict, thres, dataroot='data'):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'val']

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = pickle.load(open(ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.question_dictionary = question_dict
        self.caption_dictionary = caption_dict

        self.img_id2idx = pickle.load(
            open(os.path.join(dataroot, '%s36_imgid2idx.pkl' % name)))
        print('loading features from h5 file')
        h5_path = os.path.join(dataroot, '%s36.hdf5' % name)
        with h5py.File(h5_path, 'r') as hf:
            self.features = np.array(hf.get('image_features'))
            #self.spatials = np.array(hf.get('spatial_features'))

        self.entries = _load_dataset(dataroot, name, self.img_id2idx, thres)

        self.tokenize()
        self.tensorize()
        self.v_num = self.features.size(1)
        self.v_dim = self.features.size(2)
        #self.s_dim = self.spatials.size(2)

    def tokenize(self, max_qu_length=14, max_cap_length=18):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            qtokens = self.question_dictionary.tokenize(entry['question'], False)
            qtokens = qtokens[:max_qu_length]
            ctokens = [self.caption_dictionary.word2idx['<start>']]
            ctokens.extend(self.caption_dictionary.tokenize(entry['caption'], False))
            ctokens.append(self.caption_dictionary.word2idx['<end>'])
            ctokens = ctokens[:max_cap_length]
            if len(qtokens) < max_qu_length:
                # Note here we pad in front of the sentence
                qpadding = [self.question_dictionary.padding_idx] \
                          * (max_qu_length - len(qtokens))
                qtokens = qpadding + qtokens
            utils.assert_eq(len(qtokens), max_qu_length)
            entry['c_len'] = len(ctokens)
            if len(ctokens) < max_cap_length:
                # Note here we pad in front of the sentence
                cpadding = [self.caption_dictionary.padding_idx] \
                          * (max_cap_length - len(ctokens))
                ctokens = ctokens + cpadding
            utils.assert_eq(len(ctokens), max_cap_length)
            entry['q_token'] = qtokens
            entry['c_token'] = ctokens

    def tensorize(self):
        self.features = torch.from_numpy(self.features)
        #self.spatials = torch.from_numpy(self.spatials)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question
            caption = torch.from_numpy(np.array(entry['c_token']))
            entry['c_token'] = caption

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        features = self.features[entry['image']]
        #spatials = self.spatials[entry['image']]

        qid = entry['question_id']
        imgid = entry['image_id']
        question = entry['q_token']
        caption = entry['c_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)

        return qid, imgid, features, question, target, caption

    def __len__(self):
        return len(self.entries)
