import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, VQAFeatureDataset, VQAEDataset, VQAEVQA2Dataset, data_factory
import base_model
from train import train, evaluate
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--att_dim', type=int, default=512)
    parser.add_argument('--decode_dim', type=int, default=1024)
    parser.add_argument('--att_thr', type=float, default=0.0)
    parser.add_argument('--model', type=str, default='vqae2_newatt')
    parser.add_argument('--share_qe_dict', type=bool, default=False)
    parser.add_argument('--dataset', type=str, default='VQAE', choices=['VQAE', 'VQA2', 'VQAEVQA2'])
    parser.add_argument('--output', type=str, default='saved_models/pretrain_VQE_then_VQA_EQA_1.0')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--evaluate', type=bool, default=False)
    parser.add_argument('--parallel', type=bool, default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    vocab_source = args.dataset[-4:]

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    if args.share_qe_dict:
        qe_dict = Dictionary.load_from_file(os.path.join('data', vocab_source, 'question_explain_dictionary.pkl'))
        q_emb_file = os.path.join('data', vocab_source, 'glove6b_init_300d_question_explain.npy')
        c_emb_file = q_emb_file
    else:
        q_dict = Dictionary.load_from_file(os.path.join('data', vocab_source, 'question_dictionary.pkl'))
        c_dict = Dictionary.load_from_file(os.path.join('data', vocab_source, 'explain_dictionary.pkl'))
        q_emb_file = os.path.join('data', vocab_source, 'glove6b_init_300d_question.npy')
        c_emb_file = os.path.join('data', vocab_source, 'glove6b_init_300d_explain.npy')

    train_dset = data_factory(args.dataset, 'train', q_dict, c_dict, os.path.join('cache', vocab_source))
    eval_dset = data_factory(args.dataset, 'val', q_dict, c_dict, os.path.join('cache', vocab_source))
    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = utils.model_factory(constructor, train_dset, args.num_hid, args.att_dim, args.decode_dim).cuda()

    if args.resume or args.evaluate:
        model_path = os.path.join(args.output, 'model.pth')
        model_state = torch.load(model_path)
        model.load_state_dict(model_state)
    else:
        args.output = args.output + '_' + str(args.temperature)
        model.w_emb.init_embedding(q_emb_file)
        model.generator.init_embedding(c_emb_file)
    print('Model has {} parameters in total'.format(utils.params_count(model)))

    if args.parallel:
        model = nn.DataParallel(model).cuda()

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1)
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1)

    if args.evaluate:
        model.train(False)
        eval_score, bound = evaluate(model, eval_loader)
        print('\teval score: vqa = %.2f / eqa = %.2f / ens = %.2f'
              % (100 * eval_score[0], 100 * eval_score[1], 100 * eval_score[2]))
    else:
        train(model, train_loader, eval_loader, args.epochs, args.output, args.temperature)
