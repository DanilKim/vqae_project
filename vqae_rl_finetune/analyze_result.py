import argparse
import os
import pdb
import time
import json
import pickle
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
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--att_dim', type=int, default=512)
    parser.add_argument('--decode_dim', type=int, default=1024)
    parser.add_argument('--att_thr', type=float, default=0.0)
    parser.add_argument('--model', type=str, default='vqae2_newatt')
    parser.add_argument('--share_qe_dict', type=bool, default=False)
    parser.add_argument('--dataset', type=str, default='VQAE', choices=['VQAE', 'VQA2', 'VQAEVQA2'])
    parser.add_argument('--output', type=str, default='saved_models/VQE_VQA_EQA_VQAE_gt_vs_gen_1.0')
    parser.add_argument('--source1', type=str, default='saved_models/VQE_VQA_EQA_VQAE_gtExp_1.0')
    parser.add_argument('--source2', type=str, default='saved_models/VQE_VQA_EQA_VQAE_1.0')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()
    return args


def compare(model1, model2, dataloader, q_dict, c_dict, label2ans, output):
    gt_better  = utils.Logger(os.path.join(output, 'gt_better.txt'))
    gen_better = utils.Logger(os.path.join(output, 'gen_better.txt'))
    both_good  = utils.Logger(os.path.join(output, 'both_good.txt'))
    both_bad   = utils.Logger(os.path.join(output, 'both_bad.txt'))
    gt_cnt   = 0
    gen_cnt  = 0
    good_cnt = 0
    bad_cnt  = 0

    for i, (qid, iid, v, q, a, c, l) in enumerate(dataloader):
        batch_size = v.size(0)
        v = Variable(v, volatile=True).cuda()
        q = Variable(q, volatile=True).cuda()
        c = Variable(c, volatile=True).cuda()

        ans_logit1, exp_pred1 = model1.evaluate(v, q, c)
        ans_logit2, exp_pred2 = model2.evaluate(v, q)

        vqa_logit1 = ans_logit1[0][0]
        eqa_logit1 = ans_logit1[1][0]
        ens_logit1 = vqa_logit1 + eqa_logit1
        vqa_logit2 = ans_logit2[0][0]
        eqa_logit2 = ans_logit2[1][0]
        ens_logit2 = vqa_logit2 + eqa_logit2

        vqa_pred1 = torch.max(vqa_logit1, dim=-1)[1].data[0]
        eqa_pred1 = torch.max(eqa_logit1, dim=-1)[1].data[0]
        ens_pred1 = torch.max(ens_logit1, dim=-1)[1].data[0]
        ans_pred1 = [vqa_pred1, eqa_pred1, ens_pred1]
        vqa_pred2 = torch.max(vqa_logit2, dim=-1)[1].data[0]
        eqa_pred2 = torch.max(eqa_logit2, dim=-1)[1].data[0]
        ens_pred2 = torch.max(ens_logit2, dim=-1)[1].data[0]
        ans_pred2 = [vqa_pred2, eqa_pred2, ens_pred2]

        if a[0][ens_pred1] >= 0.6 > a[0][ens_pred2]:
            gt_cnt += 1
            logging(gt_better, gt_cnt, qid[0], iid[0], q[0], a[0], ans_pred1, ans_pred2, c[0], exp_pred2[0], q_dict, c_dict)
        elif a[0][ens_pred1] < 0.6 <= a[0][ens_pred2]:
            gen_cnt += 1
            logging(gen_better, gen_cnt, qid[0], iid[0], q[0], a[0], ans_pred1, ans_pred2, c[0], exp_pred2[0], q_dict, c_dict)
        elif a[0][ens_pred1] >= 0.6 and a[0][ens_pred2] >= 0.6:
            good_cnt += 1
            logging(both_good, good_cnt, qid[0], iid[0], q[0], a[0], ans_pred1, ans_pred2, c[0], exp_pred2[0], q_dict, c_dict)
        elif a[0][ens_pred1] < 0.6 and a[0][ens_pred2] < 0.6:
            bad_cnt += 1
            logging(both_bad, bad_cnt, qid[0], iid[0], q[0], a[0], ans_pred1, ans_pred2, c[0], exp_pred2[0], q_dict, c_dict)

def logging(logger, idx, qid, iid, q, a_gt, a1, a2, e_gt, e_gen, q_dict, c_dict):
    logger.write('%d. Image ID: %d \t Question ID: %d \t' % (idx, iid, qid))
    logger.write('Question   : ' + ' '.join([q_dict.idx2word[p] for p in q.data.tolist() if p > 0]))
    logger.write('Answer GT  : ' + label2ans[a_gt.max(dim=-1)[1][0]])
    logger.write('Explain GT : ' + ' '.join([c_dict.idx2word[p] for p in e_gt.data.tolist() if p > 2]))
    logger.write('Explain Gen: ' + ' '.join([c_dict.idx2word[p] for p in e_gen.data.tolist() if p > 2]))
    logger.write('GT  Exp Answer Pred: (Ens) - %s \t (VQA) - %s \t (EQA) - %s'
                 % (label2ans[a1[2]], label2ans[a1[0]], label2ans[a1[1]]))
    logger.write('Gen Exp Answer Pred: (Ens) - %s \t (VQA) - %s \t (EQA) - %s\n'
                 % (label2ans[a2[2]], label2ans[a2[0]], label2ans[a2[1]]))


def evaluate(model, dataloader, q_dict, c_dict, gt=True):
    score = 0
    vq_score = 0
    eq_score = 0
    print_freq = 50
    results = []
    for i, (qid, iid, v, q, a, c, l) in enumerate(dataloader):
        batch_size = v.size(0)
        v = Variable(v, volatile=True).cuda()
        q = Variable(q, volatile=True).cuda()

        if not gt:
            c = None
        ans_pred, exp_pred = model.evaluate(v, q, c)
        if type(ans_pred) is list:
            vq_batch_score = compute_score_with_logits(ans_pred[0], a.cuda()).sum()
            eq_batch_score = compute_score_with_logits(ans_pred[1], a.cuda()).sum()
            vq_score += vq_batch_score
            eq_score += eq_batch_score
            ans_pred = ans_pred[0] + ans_pred[1]

        batch_score = compute_score_with_logits(ans_pred, a.cuda()).sum()
        score += batch_score
        #batch_score = compute_score_with_logits(ans_pred, a.cuda()).sum()

        # for j in range(batch_size):
        #     if c[j][0] != 0:
        #         results.append({
        #             'question_id': qid[j],
        #             'image_id': iid[j],
        #             'question': ' '.join([q_dict.idx2word[p] for p in q[j].data.tolist() if p > 0]),
        #             'answer': dataloader.dataset.label2ans[ans_pred[0][j].max(0)[1].data[0]],
        #             'explain_res': ' '.join([c_dict.idx2word[p] for p in exp_pred[j].data.tolist() if p > 2]),
        #             'explain_gt': ' '.join([c_dict.idx2word[p] for p in c[j].tolist() if p > 2])
        #         })

    score = score / len(dataloader.dataset)

    if vq_score == 0:
        scores = score
    else:
        vq_score = vq_score / len(dataloader.dataset)
        eq_score = eq_score / len(dataloader.dataset)
        scores = {'vq_score': vq_score, 'eq_score': eq_score, 'ens_score': score}
    print('Scores :', scores)
    return scores, results


def save_results(results, savedir):
    path_rslt = os.path.join(savedir, 'results.json')
    with open(path_rslt, 'w') as handle:
        json.dump(results, handle)


if __name__ == '__main__':
    args = parse_args()
    vocab_source = args.dataset[-4:]

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    if args.share_qe_dict:
        qe_dict = Dictionary.load_from_file(os.path.join('data', vocab_source, 'question_explain_dictionary.pkl'))
    else:
        q_dict = Dictionary.load_from_file(os.path.join('data', vocab_source, 'question_dictionary.pkl'))
        c_dict = Dictionary.load_from_file(os.path.join('data', vocab_source, 'explain_dictionary.pkl'))

    train_dset = data_factory(args.dataset, 'train', q_dict, c_dict, os.path.join('cache', vocab_source))
    eval_dset = data_factory(args.dataset, 'val', q_dict, c_dict, os.path.join('cache', vocab_source))
    label2ans = pickle.load(open(os.path.join('data', 'cache', vocab_source, 'trainval_label2ans.pkl')))
    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model1 = utils.model_factory(constructor, train_dset, args.num_hid, args.att_dim, args.decode_dim).cuda()
    model2 = utils.model_factory(constructor, train_dset, args.num_hid, args.att_dim, args.decode_dim).cuda()

    model1_path = os.path.join(args.source1, 'model.pth')
    model1_state = torch.load(model1_path)
    model1.load_state_dict(model1_state)
    model2_path = os.path.join(args.source2, 'model.pth')
    model2_state = torch.load(model2_path)
    model2.load_state_dict(model2_state)

    print('Model has {} parameters in total'.format(utils.params_count(model1)))

    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=1)
    model1.train(False)
    model2.train(False)
    compare(model1, model2, eval_loader, q_dict, c_dict, label2ans, args.output)
    evaluate(model1, eval_loader, q_dict, c_dict, True)
    evaluate(model2, eval_loader, q_dict, c_dict, False)

    #vqa_score, results = evaluate(model1, eval_loader, q_dict, c_dict)
    #save_obj = {'vqa_score': vqa_score, 'results': results}
    #save_results(save_obj, args.output)
