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

from dataset import Dictionary, VQAFeatureDataset, VQAEDataset, VQAEVQA2Dataset, data_factory
import base_model
from train import compute_score_with_logits
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vqae2_newatt')
    parser.add_argument('--emb_rnn', type=str, default='GRU')
    parser.add_argument('--dim_hid', type=int, default=1024)
    parser.add_argument('--train_set', type=str, default='VQAE', choices=['VQAE', 'VQA2'])
    parser.add_argument('--val_set', type=str, default='VQAE', choices=['VQAE', 'VQA2'])
    parser.add_argument('--output', type=str, default='saved_models/VQE_VQEA_gen')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()
    return args


def evaluate(model, dataloader, q_dict, c_dict):
    total_others_score = 0
    total_yesno_score = 0
    total_number_score = 0
    total_all_score = 0
    vq_score = 0
    eq_score = 0
    print_freq = 50
    results = []

    ann_path = os.path.join('data','v2_mscoco_val2014_annotations.json')
    anns = json.load(open(ann_path))['annotations']
    qid2atype = {a['question_id'] : a['answer_type'] for a in anns}
    #qid2qtype = {a['question_id'] : a['question_type'] for a in anns}
    atype_map = {'other':0, 'yes/no':1, 'number':2}

    others_cnt = 0
    yesno_cnt = 0
    number_cnt = 0
    for i, (qid, iid, v, q, a, c, l) in enumerate(dataloader):
        batch_size = v.size(0)
        v = Variable(v, volatile=True).cuda()
        q = Variable(q, volatile=True).cuda()

        ans_pred, exp_pred = model.evaluate(v, q)

        atype_list = torch.Tensor([atype_map[qid2atype[id]] for id in qid])
        others_idx = (atype_list==0).nonzero().squeeze()
        yesno_idx = (atype_list==1).nonzero().squeeze()
        number_idx = (atype_list==2).nonzero().squeeze()

        others_cnt += len(others_idx)
        yesno_cnt += len(yesno_idx)
        number_cnt += len(number_idx)

        others_score = compute_score_with_logits(ans_pred[others_idx.cuda()], a[others_idx].cuda()).sum()
        yesno_score = compute_score_with_logits(ans_pred[yesno_idx.cuda()], a[yesno_idx].cuda()).sum()
        number_score = compute_score_with_logits(ans_pred[number_idx.cuda()], a[number_idx].cuda()).sum()

        total_others_score += others_score
        total_yesno_score += yesno_score
        total_number_score += number_score

        all_score = compute_score_with_logits(ans_pred, a.cuda()).sum()
        total_all_score += all_score

        for j in range(batch_size):
            results.append({
                'question_id': qid[j],
                'image_id': iid[j],
                'question': ' '.join([q_dict.idx2word[p] for p in q[j].data.tolist() if p > 0]),
                'answer_pred': dataloader.dataset.label2ans[ans_pred[j].max(0)[1].data[0]],
                'answer_gt': dataloader.dataset.label2ans[int(a[j].max(0)[1])],
                'explain_res': ' '.join([c_dict.idx2word[p] for p in exp_pred[j].data.tolist() if p > 2]),
                'explain_gt': ' '.join([c_dict.idx2word[p] for p in c[j].tolist() if p > 2])
            })

    results = sorted(results, key=lambda x: x['question_id'])
    total_all_score = total_all_score / len(dataloader.dataset)
    total_others_score /= others_cnt
    total_yesno_score /= yesno_cnt
    total_number_score /= number_cnt

    scores = {'all':total_all_score, 'others':total_others_score, 'yes/no':total_yesno_score, 'number':total_number_score}
    return scores, results


def save_results(results, savedir):
    path_rslt = os.path.join(savedir, 'results.json')
    with open(path_rslt, 'w') as handle:
        json.dump(results, handle)

if __name__ == '__main__':
    args = parse_args()
    vocab_source = 'VQA2'  #### 'VQAE' or 'VQAv2'
    target_source = vocab_source + '_2_' + args.val_set

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    q_dict = Dictionary.load_from_file(os.path.join('data', vocab_source, 'question_dictionary.pkl'))
    c_dict = Dictionary.load_from_file(os.path.join('data', vocab_source, 'explain_dictionary.pkl'))
    q_emb_file = os.path.join('data', vocab_source, 'glove6b_init_300d_question.npy')
    c_emb_file = os.path.join('data', vocab_source, 'glove6b_init_300d_explain.npy')

    train_dset = data_factory(args.train_set, 'train', q_dict, c_dict, os.path.join('cache', target_source), result=True)
    eval_dset = data_factory(args.val_set, 'val', q_dict, c_dict, os.path.join('cache', target_source), result=True)
    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = utils.model_factory(constructor, train_dset, args.dim_hid, args.emb_rnn).cuda()

    model_path = os.path.join(args.output, 'model.pth')
    model_state = torch.load(model_path)
    model.load_state_dict(model_state)

    print('Model has {} parameters in total'.format(utils.params_count(model)))
    #model = nn.DataParallel(model).cuda()

    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=1)
    model.train(False)
    vqa_score, results = evaluate(model, eval_loader, q_dict, c_dict)
    print(vqa_score)
    save_obj = {'vqa_score': vqa_score, 'results': results}
    import pdb
    pdb.set_trace()
    save_results(save_obj, args.output)
    # VQAave_results(results, args.output)
