import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, VQAFeatureDataset, VQAEDataset, VQAXDataset, VQAEVQA2Dataset, data_factory
import base_model
from train import train, evaluate
import train_base
import utils
import pdb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--att_dim', type=int, default=512)
    parser.add_argument('--decode_dim', type=int, default=1024)
    parser.add_argument('--model', type=str, default='generator', choices=['generator', 'answer_predictor'])
    parser.add_argument('--share_qe_dict', type=bool, default=False)
    parser.add_argument('--train_set', type=str, default='VQAE', choices=['VQAE', 'VQAX', 'VQA2', 'VQAEVQA2'])
    parser.add_argument('--val_set', type=str, default='VQAE', choices=['VQAE', 'VQAX', 'VQA2', 'VQAEVQA2'])
    parser.add_argument('--output', type=str, default='saved_models/')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--evaluate', type=bool, default=False)
    parser.add_argument('--parallel', type=bool, default=False)
    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--trial_gen', type=str, default='VQE_visualize_train')
    parser.add_argument('--trial_vqa', type=str, default='GRUCell_last_hidden')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.model == 'generator':
        from train_generator import train, evaluate
        args.model = 'vqe_newatt'
        args.model_fn = 'generator.pth'
    elif args.model == 'answer_predictor':
        from train_vqa import train, evaluate
        args.model = 'lstm_vqa'
        args.model_fn = 'answer_predictor_%s.pth' % args.trial_vqa
    else:
        ValueError

    vocab_source = 'VQA2' #args.train_set[-4:]
    target_source = vocab_source + '_2_' + args.val_set
    args.output = args.output + 'VQAE' #args.train_set
    if args.debug:
        args.output = args.output + '_' + str(args.trial_gen)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    if args.share_qe_dict:
        qe_dict = Dictionary.load_from_file(os.path.join('data', vocab_source, 'question_explain_dictionary.pkl'))
        q_emb_file = os.path.join('data', vocab_source, 'glove6b_init_300d_question_explain.npy')
        c_emb_file = q_emb_file
    else:
        q_dict = Dictionary.load_from_file(os.path.join('data', args.train_set, 'question_dictionary.pkl'))
        c_dict = Dictionary.load_from_file(os.path.join('data', args.train_set, 'explain_dictionary.pkl'))
        q_emb_file = os.path.join('data', args.train_set, 'glove6b_init_300d_question.npy')
        c_emb_file = os.path.join('data', args.train_set, 'glove6b_init_300d_explain.npy')

    if args.evaluate:
        args.train_set = args.val_set

    train_dset = data_factory(args.train_set, 'train', q_dict, c_dict, os.path.join('cache', target_source))
    eval_dset = data_factory(args.val_set, 'val', q_dict, c_dict, os.path.join('cache', target_source))
    train_loader = DataLoader(train_dset, args.batch_size, shuffle=True, num_workers=1)
    eval_loader =  DataLoader(eval_dset, args.batch_size, shuffle=True, num_workers=1)

    constructor = 'build_%s' % args.model
    model = utils.model_factory(constructor, train_dset, args.num_hid, args.att_dim, args.decode_dim).cuda()

    if args.resume or args.evaluate:
        model_path = os.path.join(args.output, args.model_fn)
        model_state = torch.load(model_path)
        model.load_state_dict(model_state)
    elif args.pretrained is not None:
        pretrained_path = os.path.join(args.pretrained, args.model_fn)
        pretrained_dict = torch.load(pretrained_path)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    elif args.model == 'answer_predictor':
        generator_path = os.path.join(args.output, 'generator.pth')
        generator_state = torch.load(generator_path)
        model.generator.load_state_dict(generator_state)
        model.w_emb.init_embedding(q_emb_file)
    elif args.model == 'generator':
        model.w_emb.init_embedding(q_emb_file)
    else:
        ValueError

    print('Model has {} parameters in total'.format(utils.params_count(model)))

    if args.parallel:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        model.train(False)
        evaluate(model, eval_loader, args.val_set)
    else:
        train(model, train_loader, eval_loader, args.epochs, args.output, args.model_fn)
