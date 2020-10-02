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
    labeled_set = []
    unlabeled_set = []
    for question, answer in zip(questions, answers):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        if explains != [] and question['question_id'] == explains[0]['question_id']:
            labeled_set.append(_create_entry(img_id2val[img_id], question, answer,
                                         explains.pop(0)['explanation'][0]))
        else:
            unlabeled_set.append(_create_entry(img_id2val[img_id], question, answer, None))

    return labeled_set, unlabeled_set


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

    answer_path = os.path.join(dataroot, answer_dir, '%s_target.pkl' % name)
    answers = pickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])

    utils.assert_eq(len(questions), len(answers))
    entries = []
    for question, answer in zip(questions, answers):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        entries.append(_create_entry(img_id2val[img_id], question, answer, None))

    return entries


class VQASemiDataset(Dataset):
    def __init__(self, name, question_dict, caption_dict, answer_dir, dataroot='data'):
        super(VQASemiDataset, self).__init__()
        assert name in ['train', 'val']

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
            #self.spatials = np.array(hf.get('spatial_features'))

        self.labeled_set, self.unlabeled_set = _load_VQA2E(dataroot, name, answer_dir, self.img_id2idx)
        self.unlabeled_cnt = len(self.unlabeled_set)
        self.shuffle_unlabeled = torch.randperm(self.unlabeled_cnt)

        self.tokenize()
        self.tensorize(self.labeled_set)
        self.tensorize(self.unlabeled_set)
        self.v_num = self.features.size(1)
        self.v_dim = self.features.size(2)
        #self.s_dim = self.spatials.size(2)

    def tokenize(self, max_qu_length=14, max_cap_length=18):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.labeled_set:
            qtokens = self.question_dictionary.tokenize(entry['question'], False)
            qtokens = qtokens[:max_qu_length]
            ctokens = [self.explanation_dictionary.word2idx['<start>']]
            ctokens.extend(self.explanation_dictionary.tokenize(entry['explanation'], False))
            ctokens.append(self.explanation_dictionary.word2idx['<end>'])
            ctokens = ctokens[:max_cap_length]
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

        for entry in self.unlabeled_set:
            qtokens = self.question_dictionary.tokenize(entry['question'], False)
            qtokens = qtokens[:max_qu_length]
            if len(qtokens) < max_qu_length:
                # Note here we pad in front of the sentence
                qpadding = [self.question_dictionary.padding_idx] \
                          * (max_qu_length - len(qtokens))
                qtokens = qpadding + qtokens
            utils.assert_eq(len(qtokens), max_qu_length)
            utils.assert_eq(len(ctokens), max_cap_length)
            entry['q_token'] = qtokens

    def tensorize(self, entries):
        if isinstance(self.features, np.ndarray):
            self.features = torch.from_numpy(self.features)
            #self.spatials = torch.from_numpy(self.spatials)

        for entry in entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question
            if 'c_token' in entry:
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
        u_index = self.shuffle_unlabeled[index % len(self.unlabeled_set)]
        L_entry = self.labeled_set[index % len(self.labeled_set)]
        UL_entry = self.unlabeled_set[u_index]

        L = {}; UL = {}
        L['features'] = self.features[L_entry['image']]
        UL['features'] = self.features[UL_entry['image']]

        L['qid'] = L_entry['question_id']
        L['iid'] = L_entry['image_id']
        L['question'] = L_entry['q_token']
        L['explain'] = L_entry['c_token']
        L['exp_len'] = L_entry['c_len']
        answer = L_entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        L['target'] = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            L['target'].scatter_(0, labels, scores)

        UL['qid'] = UL_entry['question_id']
        UL['iid'] = UL_entry['image_id']
        UL['question'] = UL_entry['q_token']
        answer = UL_entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        UL['target'] = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            UL['target'].scatter_(0, labels, scores)

        return L, UL

    def __len__(self):
        return max(len(self.labeled_set), len(self.unlabeled_set))


class VQAFullDataset(VQASemiDataset):
    def __init__(self, name, question_dict, caption_dict, answer_dir, dataroot='data'):
        super(VQAFullDataset, self).__init__(name, question_dict, caption_dict, answer_dir, dataroot)
        for entry in self.unlabeled_set:
            entry['c_token'] = torch.LongTensor([0]*18)
            entry['c_len'] = 0

        self.entries = self.labeled_set + self.unlabeled_set

    def __getitem__(self, index):
        entry = self.entries[index]

        Entry = {}
        Entry['features'] = self.features[entry['image']]

        Entry['qid'] = entry['question_id']
        Entry['iid'] = entry['image_id']
        Entry['question'] = entry['q_token']
        Entry['explain'] = entry['c_token']
        Entry['exp_len'] = entry['c_len']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        Entry['target'] = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            Entry['target'].scatter_(0, labels, scores)

        return Entry

    def __len__(self):
        return len(self.entries)


class VQADataset(Dataset):
    def __init__(self, name, question_dict, explanation_dict, answer_dir, dataroot='data', result=False):
        super(VQADataset, self).__init__()
        assert name in ['train', 'val']
        self.result = result

        ans2label_path = os.path.join(dataroot, answer_dir, 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, answer_dir, 'trainval_label2ans.pkl')
        self.ans2label = pickle.load(open(ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.question_dictionary = question_dict

        self.img_id2idx = pickle.load(
            open(os.path.join(dataroot, '%s36_imgid2idx.pkl' % name)))
        self.num_images = len(self.img_id2idx)
        print('loading features from h5 file')
        h5_path = os.path.join(dataroot, '%s36.hdf5' % name)
        with h5py.File(h5_path, 'r') as hf:
            self.features = np.array(hf.get('image_features'))
            #self.spatials = np.array(hf.get('spatial_features'))

        self.entries = _load_dataset(dataroot, name, answer_dir, self.img_id2idx)

        self.tokenize()
        self.tensorize()
        self.v_num = self.features.size(1)
        self.v_dim = self.features.size(2)
        #self.s_dim = self.spatials.size(2)

    def tokenize(self, max_qu_length=14, max_exp_length=18):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            qtokens = self.question_dictionary.tokenize(entry['question'], False)
            qtokens = qtokens[:max_qu_length]
            if len(qtokens) < max_qu_length:
                # Note here we pad in front of the sentence
                qpadding = [self.question_dictionary.padding_idx] \
                          * (max_qu_length - len(qtokens))
                qtokens = qpadding + qtokens
            utils.assert_eq(len(qtokens), max_qu_length)
            entry['q_token'] = qtokens

    def tensorize(self):
        self.features = torch.from_numpy(self.features)
        #self.spatials = torch.from_numpy(self.spatials)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

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
        Entry = {}

        Entry['features'] = self.features[entry['image']]
        Entry['qid'] = entry['question_id']
        Entry['iid'] = entry['image_id']
        Entry['question'] = entry['q_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        Entry['target'] = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            Entry['target'].scatter_(0, labels, scores)

        return Entry

    def __len__(self):
        return len(self.entries)


class VQAEDataset(Dataset):
    def __init__(self, name, question_dict, explanation_dict, answer_dir, dataroot='data', result=False):
        super(VQAEDataset, self).__init__()
        assert name in ['train', 'val']
        self.result = result

        ans2label_path = os.path.join(dataroot, answer_dir, 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, answer_dir, 'trainval_label2ans.pkl')
        self.ans2label = pickle.load(open(ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.question_dictionary = question_dict
        self.explanation_dictionary = explanation_dict

        self.img_id2idx = pickle.load(
            open(os.path.join(dataroot, '%s36_imgid2idx.pkl' % name)))
        self.num_images = len(self.img_id2idx)
        print('loading features from h5 file')
        h5_path = os.path.join(dataroot, '%s36.hdf5' % name)
        with h5py.File(h5_path, 'r') as hf:
            self.features = np.array(hf.get('image_features'))
            #self.spatials = np.array(hf.get('spatial_features'))

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
                                 'fake_image'  : (self.img_id2idx[img_id] + 1) % self.num_images,
                                 'question'    : explain['question'],
                                 'answer'      : answer,
                                 'explanation' : explain['explanation'][0]})

        self.tokenize()
        self.tensorize()
        self.v_num = self.features.size(1)
        self.v_dim = self.features.size(2)
        #self.s_dim = self.spatials.size(2)

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
        #self.spatials = torch.from_numpy(self.spatials)

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
        #fake_feat = self.features[entry['fake_image']]

        Entry = {}
        Entry['features'] = self.features[entry['image']]

        Entry['qid'] = entry['question_id']
        Entry['iid'] = entry['image_id']
        Entry['question'] = entry['q_token']
        Entry['explain'] = entry['e_token']
        Entry['exp_len'] = entry['e_len']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        Entry['target'] = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            Entry['target'].scatter_(0, labels, scores)

        return Entry

    def __len__(self):
        return len(self.entries)


def data_factory(data, split, q_dict, c_dict, t_dir, result=False):
    if data == 'VQAE':
        dset = VQAEDataset(split, q_dict, c_dict, t_dir, result=result)
    elif data == 'VQAS':
        dset = VQASemiDataset(split, q_dict, c_dict, t_dir)
    elif data == 'VQAF':
        dset = VQAFullDataset(split, q_dict, c_dict, t_dir)
    elif data == 'VQA2':
        dset = VQADataset(split, q_dict, c_dict, t_dir)
    else:
        raise ValueError

    return dset
