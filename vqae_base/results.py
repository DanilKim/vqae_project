import argparse
import os
import pdb
import time
import json
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, VQAFeatureDataset, VQAEDataset, VQAEVQA2Dataset
import base_model
from train import compute_score_with_logits
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--att_dim', type=int, default=512)
    parser.add_argument('--decode_dim', type=int, default=1024)
    parser.add_argument('--att_thr', type=float, default=0.0)
    parser.add_argument('--model', type=str, default='vqae_newatt')
    parser.add_argument('--output', type=str, default='saved_models/QIA100E1_VQAE_VQAE_Adamax_2e-3_nVQA_ST_1024')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()
    return args


def evaluate(model, dataloader, q_dict, c_dict):
    score = 0
    upper_bound = 0
    num_data = 0
    print_freq = 50
    results = []
    for i, (qid, iid, v, q, a, c, l) in enumerate(dataloader):
        batch_size = v.size(0)
        v = Variable(v, volatile=True).cuda()
        q = Variable(q, volatile=True).cuda()

        ans_pred, exp_pred = model.evaluate(v, q)
        batch_score = compute_score_with_logits(ans_pred, a.cuda()).sum()
        score += batch_score
        for j in range(batch_size):
            if c[j][0] != 0:
                results.append({
                    'question_id': qid[j],
                    'image_id': iid[j],
                    'question': ' '.join([q_dict.idx2word[p] for p in q[j].data.tolist() if p > 0]),
                    'answer': dataloader.dataset.label2ans[ans_pred[j].max(0)[1].data[0]],
                    'explain_res': ' '.join([c_dict.idx2word[p] for p in exp_pred[j].data.tolist() if p > 2]),
                    'explain_gt': ' '.join([c_dict.idx2word[p] for p in c[j].tolist() if p > 2])
                })

        if False: #i % print_freq == 0:
            print('Batch: [%d/%d]\nQuestion: %s\t Answer Prediction: %s\n'
                  'Explain GT: %s\nExplain Prediction: %s'%
                  (i, len(dataloader), results[i * j]['question'], results[i * j]['answer'],
                   results[i * j]['explain_gt'], results[i * j]['explain_res']))

    score = score / len(dataloader.dataset)
    return score, results


def save_results(results, savedir):
    path_rslt = os.path.join(savedir, 'results.json')
    with open(path_rslt, 'w') as handle:
        json.dump(results, handle)


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    #q_dict = Dictionary.load_from_file('data/question_dictionary.pkl')
    #c_dict = Dictionary.load_from_file('data/caption_dictionary.pkl')
    q_dict = Dictionary.load_from_file('data/VQAE/question_dictionary.pkl')
    c_dict = Dictionary.load_from_file('data/VQAE/explain_dictionary.pkl')

    #train_dset = VQAFeatureDataset('train', q_dict, c_dict, args.att_thr)
    #eval_dset = VQAFeatureDataset('val', q_dict, c_dict, args.att_thr)
    train_dset = VQAEDataset('train', q_dict, c_dict, 'cache/VQAE2')
    eval_dset = VQAEDataset('val', q_dict, c_dict, 'cache/VQAE2')
    #train_dset = VQAEVQA2Dataset('train', q_dict, c_dict, 'cache')
    #eval_dset = VQAEVQA2Dataset('val', q_dict, c_dict, 'cache')
    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = utils.factory(constructor, train_dset, args.num_hid, args.att_dim, args.decode_dim).cuda()

    model_path = os.path.join(args.output, 'model.pth')
    model_state = torch.load(model_path)
    model.load_state_dict(model_state)

    print('Model has {} parameters in total'.format(utils.params_count(model)))
    #model = nn.DataParallel(model).cuda()

    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=1)
    model.train(False)
    vqa_score, results = evaluate(model, eval_loader, q_dict, c_dict)
    save_obj = {'vqa_score': vqa_score, 'results': results}
    save_results(save_obj, args.output)
    #save_results(results, args.output)
