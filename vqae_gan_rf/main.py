import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pdb

from dataset import Dictionary, VQADataset, VQASemiDataset, VQAEDataset, VQAFullDataset, data_factory
import base_model
from train import train, evaluate
from pretrain import pretrain_ans, pretrain_gen
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--att_dim', type=int, default=512)
    parser.add_argument('--decode_dim', type=int, default=1024)
    parser.add_argument('--model', type=str, default='newatt_2')
    parser.add_argument('--share_qe_dict', type=bool, default=False)
    parser.add_argument('--train_set', type=str, default='VQAS', choices=['VQAE', 'VQAS', 'VQAF', 'VQA2'])
    parser.add_argument('--val_set', type=str, default='VQAF', choices=['VQAE', 'VQAS', 'VQAF', 'VQA2'])
    parser.add_argument('--output', type=str, default='saved_models/semi_pAE30_Enc_gt_G1D0_VQAE')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--pretrained', type=str, default='pretrained_vqae_enc_gt.pth')
    parser.add_argument('--evaluate', type=bool, default=False)
    parser.add_argument('--parallel', type=bool, default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    vocab_source = 'VQA2' #args.train_set[-4:]
    target_source = 'VQA2_2_VQA2' #vocab_source + '_2_' + args.val_set

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

    if args.evaluate:
        args.train_set = args.val_set

    train_dset = data_factory(args.train_set, 'train', q_dict, c_dict, os.path.join('cache', target_source))
    eval_dset = data_factory(args.val_set, 'val', q_dict, c_dict, os.path.join('cache', target_source))
    batch_size = args.batch_size

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1)
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1)

    constructor = 'build_vqae_%s' % args.model
    model = utils.model_factory(constructor, train_dset, args.num_hid, args.att_dim, args.decode_dim).cuda()

    if args.resume:# or args.evaluate:
        model_path = os.path.join(args.output, 'model.pth')
        model_state = torch.load(model_path)
        model.load_state_dict(model_state)
    elif args.pretrained is not None:
        print('Loading VQA & VQE pretrained model...')
        pretrained_path = os.path.join(args.output, args.pretrained)
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

    if args.evaluate:
        model.train(False)
        eval_score, bound = evaluate(model, eval_loader, 0, [0], utils.Logger(os.path.join(args.output, 'eval_pt_vqae.txt')))
        print('\teval score: vqa = %.2f' % (100 * eval_score))
    else:
        train(model, train_loader, eval_loader, args.epochs, args.output)
