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
import pdb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--dim_hid', type=int, default=1024)
    parser.add_argument('--emb_rnn', type=str, default='GRU')
    parser.add_argument('--model', type=str, default='vqae2_newatt')
    parser.add_argument('--train_set', type=str, default='VQA2', choices=['VQAE', 'VQA2', 'VQAEVQA2'])
    parser.add_argument('--val_set', type=str, default='VQA2', choices=['VQAE', 'VQA2', 'VQAEVQA2'])
    parser.add_argument('--output', type=str, default='saved_models/VQAE')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--evaluate', type=bool, default=False)
    parser.add_argument('--parallel', type=bool, default=False)
    parser.add_argument('--debug', type=bool, default=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    vocab_source = 'VQA2' #args.train_set[-4:]
    target_source = vocab_source + '_2_' + args.val_set

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    q_dict = Dictionary.load_from_file(os.path.join('data', vocab_source, 'question_dictionary.pkl'))
    c_dict = Dictionary.load_from_file(os.path.join('data', vocab_source, 'explain_dictionary.pkl'))
    q_emb_file = os.path.join('data', vocab_source, 'glove6b_init_300d_question.npy')
    c_emb_file = os.path.join('data', vocab_source, 'glove6b_init_300d_explain.npy')

    if args.evaluate:
        args.train_set = args.val_set

    train_dset = data_factory(args.val_set, 'train', q_dict, c_dict, os.path.join('cache', target_source))
    eval_dset = data_factory(args.val_set, 'val', q_dict, c_dict, os.path.join('cache', target_source))
    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = utils.model_factory(constructor, train_dset, args.dim_hid, args.emb_rnn).cuda()

    if args.resume or args.evaluate:
        model_path = os.path.join(args.output, 'model.pth')
        model_state = torch.load(model_path)
        model.load_state_dict(model_state)
    elif args.pretrained is not None:
        pretrained_path = os.path.join(args.pretrained, 'model.pth')
        pretrained_dict = torch.load(pretrained_path)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        model.w_emb.init_embedding(q_emb_file)
        model.generator.init_embedding(c_emb_file)
    print('Model has {} parameters in total'.format(utils.params_count(model)))

    if args.parallel:
        model = nn.DataParallel(model).cuda()

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1)
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=False, num_workers=1)

    if args.evaluate:
        model.train(False)
        eval_score, bound = evaluate(model, eval_loader, args.val_set)
        print('\teval score: vqa = %.2f' % (100 * eval_score))
    else:
        train(model, train_loader, eval_loader, args.epochs, args.output, args.train_set, args.temperature)
