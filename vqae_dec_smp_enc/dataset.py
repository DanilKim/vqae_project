from __future__ import print_function
import os
import json
import pickle
import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset
import pdb


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


def _create_entry(img, question, answer, explanation):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer,
        'explanation' : explanation}
    return entry


def _load_VQA2E(dataroot, name, answer_dir, img_id2val):
    question_path = os.path.join(
        dataroot, 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])

    q2qid = {(q['image_id'], q['question']): [] for q in questions}
    for q in questions:
        q2qid[(q['image_id'], q['question'])].append(q['question_id'])

    explain_path = os.path.join(dataroot, 'VQA-E_%s_set.json' % name)
    explains = json.load(open(explain_path))

    for ex in explains:
        ex['question_id'] = q2qid[(ex['img_id'], ex['question'])].pop(0)
    explains = sorted(explains, key=lambda x: x['question_id'])

    answer_path = os.path.join(dataroot, answer_dir, '%s_target.pkl' % name)   # VQA on VQAE vocab
    answers = pickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])

    utils.assert_eq(len(questions), len(answers))
    entries = []
    for question, answer in zip(questions, answers):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        if explains != [] and question['question_id'] == explains[0]['question_id']:
            entries.append(_create_entry(img_id2val[img_id], question, answer,
                                         explains.pop(0)['explanation'][0]))
        else:
            entries.append(_create_entry(img_id2val[img_id], question, answer, None))

    return entries


def _load_dataset(dataroot, name, answer_dir, img_id2val):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    question_path = os.path.join(
        dataroot, 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])

    from nltk.tokenize import word_tokenize

    answer_path = os.path.join(dataroot, answer_dir, '%s_target.pkl' % name)
    answers = pickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])

    utils.assert_eq(len(questions), len(answers))
    entries = []
    for question, answer in zip(questions, answers):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        entries.append(_create_entry(img_id2val[img_id], question, answer,
                                         'this is d1ummy caption'))

    return entries


class VQAFeatureDataset(Dataset):
    def __init__(self, name, question_dict, caption_dict, answer_dir, result=False):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'val']
        self.result = result
        dataroot = 'data'

        ans2label_path = os.path.join(dataroot, answer_dir, 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, answer_dir, 'trainval_label2ans.pkl')
        self.ans2label = pickle.load(open(ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.question_dictionary = question_dict
        self.explanation_dictionary = caption_dict

        self.img_id2idx = pickle.load(
            open(os.path.join(dataroot, '%s36_imgid2idx.pkl' % name)))
        print('loading features from h5 file')
        h5_path = os.path.join(dataroot, '%s36.hdf5' % name)
        with h5py.File(h5_path, 'r') as hf:
            self.features = np.array(hf.get('image_features'))

        self.entries = _load_VQA2E(dataroot, name, answer_dir, self.img_id2idx)

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
            if entry['explanation'] is not None:
                ctokens = [self.explanation_dictionary.word2idx['<start>']]
                ctokens.extend(self.explanation_dictionary.tokenize(entry['explanation'], False))
                ctokens.append(self.explanation_dictionary.word2idx['<end>'])
                ctokens = ctokens[:max_cap_length]
                entry['loss_flag'] = 1
            else:
                ctokens = [0] * max_cap_length
                entry['loss_flag'] = 0
            if len(qtokens) < max_qu_length:
                # Note here we pad in front of the sentence
                qpadding = [self.question_dictionary.padding_idx] \
                          * (max_qu_length - len(qtokens))
                qtokens = qpadding + qtokens
            utils.assert_eq(len(qtokens), max_qu_length)
            entry['c_len'] = len(ctokens)
            if len(ctokens) < max_cap_length:
                # Note here we pad behind the sentence
                cpadding = [self.explanation_dictionary.padding_idx] \
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
        iid = entry['image_id']
        question = entry['q_token']
        caption = entry['c_token']
        cap_len = entry['c_len']
        answer = entry['answer']
        loss_flag = entry['loss_flag']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)

        if self.result:
            return qid, iid, features, question, target, caption, cap_len
        else:
            return features, question, target, caption, cap_len, loss_flag

    def __len__(self):
        return len(self.entries)


class VQAEVQA2Dataset(Dataset):
    def __init__(self, name, question_explain_dict, answer_dir, result=False):
        super(VQAEVQA2Dataset, self).__init__()
        assert name in ['train', 'val']
        self.result = result
        dataroot = 'data'

        ans2label_path = os.path.join(dataroot, answer_dir, 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, answer_dir, 'trainval_label2ans.pkl')
        self.ans2label = pickle.load(open(ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.question_explain_dictionary = question_explain_dict

        self.img_id2idx = pickle.load(
            open(os.path.join(dataroot, '%s36_imgid2idx.pkl' % name)))
        print('loading features from h5 file')
        h5_path = os.path.join(dataroot, '%s36.hdf5' % name)
        with h5py.File(h5_path, 'r') as hf:
            self.features = np.array(hf.get('image_features'))
            #self.spatials = np.array(hf.get('spatial_features'))

        self.entries = _load_VQA2E(dataroot, name, answer_dir, self.img_id2idx)

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
            qtokens = self.question_explain_dictionary.tokenize(entry['question'], False)
            qtokens = qtokens[:max_qu_length]
            if entry['explanation'] is not None:
                ctokens = [self.question_explain_dictionary.word2idx['<start>']]
                ctokens.extend(self.question_explain_dictionary.tokenize(entry['explanation'], False))
                ctokens.append(self.question_explain_dictionary.word2idx['<end>'])
                ctokens = ctokens[:max_cap_length]
                entry['loss_flag'] = 1
            else:
                ctokens = [0] * max_cap_length
                entry['loss_flag'] = 0
            if len(qtokens) < max_qu_length:
                # Note here we pad in front of the sentence
                qpadding = [self.question_explain_dictionary.padding_idx] \
                          * (max_qu_length - len(qtokens))
                qtokens = qpadding + qtokens
            utils.assert_eq(len(qtokens), max_qu_length)
            entry['c_len'] = len(ctokens)
            if len(ctokens) < max_cap_length:
                # Note here we pad behind the sentence
                cpadding = [self.question_explain_dictionary.padding_idx] \
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
        iid = entry['image_id']
        question = entry['q_token']
        caption = entry['c_token']
        cap_len = entry['c_len']
        answer = entry['answer']
        loss_flag = entry['loss_flag']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)

        if self.result:
            return qid, iid, features, question, target, caption, cap_len
        else:
            return features, question, target, caption, cap_len, loss_flag

    def __len__(self):
        return len(self.entries)


class VQAEDataset(Dataset):
    def __init__(self, name, question_dict, explanation_dict, answer_dir, result=False):
        super(VQAEDataset, self).__init__()
        assert name in ['train', 'val']
        self.result = result
        dataroot = 'data'

        ans2label_path = os.path.join(dataroot, answer_dir, 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, answer_dir, 'trainval_label2ans.pkl')
        self.ans2label = pickle.load(open(ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.question_dictionary = question_dict
        self.explanation_dictionary = explanation_dict

        self.img_id2idx = pickle.load(
            open(os.path.join(dataroot, '%s36_imgid2idx.pkl' % name)))
        print('loading features from h5 file')
        h5_path = os.path.join(dataroot, '%s36.hdf5' % name)
        with h5py.File(h5_path, 'r') as hf:
            self.features = np.array(hf.get('image_features'))

        question_path = os.path.join(
            dataroot, 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
        questions = sorted(json.load(open(question_path))['questions'],
                           key=lambda x: x['question_id'])

        q2qid = {(q['image_id'], q['question']): [] for q in questions}
        for q in questions:
            q2qid[(q['image_id'], q['question'])].append(q['question_id'])

        explain_path = os.path.join(dataroot, 'VQA-E_%s_set.json' % name)
        explains = json.load(open(explain_path))

        for ex in explains:
            ex['question_id'] = q2qid[(ex['img_id'], ex['question'])].pop(0)
        explains = sorted(explains, key=lambda x:x['question_id'])

        answer_path = os.path.join(dataroot, answer_dir, '%s_target.pkl' % name)
        answers = pickle.load(open(answer_path, 'rb'))
        answers = sorted(answers, key=lambda x: x['question_id'])

        utils.assert_eq(len(explains), len(answers))
        self.entries = []
        for explain, answer in zip(explains, answers):
            utils.assert_eq(explain['question_id'], answer['question_id'])
            utils.assert_eq(explain['img_id'], answer['image_id'])
            img_id = explain['img_id']
            self.entries.append({'question_id' : answer['question_id'],
                                 'image_id'    : answer['image_id'],
                                 'image'       : self.img_id2idx[img_id],
                                 'question'    : explain['question'],
                                 'answer'      : answer,
                                 'explanation' : explain['explanation'][0]})

        self.tokenize()
        self.tensorize()
        self.v_num = self.features.size(1)
        self.v_dim = self.features.size(2)

    def tokenize(self, max_qu_length=14, max_exp_length=18):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            qtokens = self.question_dictionary.tokenize(entry['question'], False)
            qtokens = qtokens[:max_qu_length]
            etokens = [self.explanation_dictionary.word2idx['<start>']]
            etokens.extend(self.explanation_dictionary.tokenize(entry['explanation'], False))
            etokens.append(self.explanation_dictionary.word2idx['<end>'])
            etokens = etokens[:max_exp_length]
            if len(qtokens) < max_qu_length:
                # Note here we pad in front of the sentence
                qpadding = [self.question_dictionary.padding_idx] \
                          * (max_qu_length - len(qtokens))
                qtokens = qpadding + qtokens
            utils.assert_eq(len(qtokens), max_qu_length)
            entry['e_len'] = len(etokens)
            if len(etokens) < max_exp_length:
                # Note here we pad behind the sentence
                epadding = [self.explanation_dictionary.padding_idx] \
                          * (max_exp_length - len(etokens))
                etokens = etokens + epadding
            utils.assert_eq(len(etokens), max_exp_length)
            entry['q_token'] = qtokens
            entry['e_token'] = etokens

    def tensorize(self):
        self.features = torch.from_numpy(self.features)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question
            explanation = torch.from_numpy(np.array(entry['e_token']))
            entry['e_token'] = explanation

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

        qid = entry['question_id']
        iid = entry['image_id']
        question = entry['q_token']
        explanation = entry['e_token']
        exp_len = entry['e_len']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)

        if self.result:
            return qid, iid, features, question, target, explanation, exp_len
        else:
            return features, question, target, explanation, exp_len, 0

    def __len__(self):
        return len(self.entries)


def data_factory(data, split, q_dict, c_dict, t_dir, result=False):
    if data == 'VQAE':
        dset = VQAEDataset(split, q_dict, c_dict, t_dir, result)
    elif data == 'VQA2':
        dset = VQAFeatureDataset(split, q_dict, c_dict, t_dir, result)
    elif data == 'VQAEVQA2':
        dset = VQAEVQA2Dataset(split, q_dict, c_dict, t_dir, result)
    else:
        raise ValueError

    return dset
