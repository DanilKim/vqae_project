import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, VQAFeatureDataset, VQAEDataset, VQAEVQA2Dataset
import base_model
from train import train, evaluate
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--att_dim', type=int, default=512)
    parser.add_argument('--decode_dim', type=int, default=1024)
    parser.add_argument('--att_thr', type=float, default=0.0)
    parser.add_argument('--model', type=str, default='vqae_newatt')
    parser.add_argument('--output', type=str, default='saved_models/pretrain_VQE_5_then_VQA_45')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--evaluate', type=bool, default=False)
    parser.add_argument('--parallel', type=bool, default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    #q_dict = Dictionary.load_from_file('data/question_dictionary.pkl')
    ###c_dict = Dictionary.load_from_file('data/caption_dictionary.pkl')
    #qe_dict = Dictionary.load_from_file('data/question_explain_dictionary.pkl')
    q_dict = Dictionary.load_from_file('data/VQAE/question_dictionary.pkl')
    c_dict = Dictionary.load_from_file('data/VQAE/explain_dictionary.pkl')
    #qe_dict = Dictionary.load_from_file('data/VQAE/question_explain_dictionary.pkl')

    #train_dset = VQAFeatureDataset('train', q_dict, c_dict, 'cache/VQAE2', args.att_thr)
    #eval_dset = VQAFeatureDataset('val', q_dict, c_dict, 'cache/VQAE2', args.att_thr)
    train_dset = VQAEDataset('train', q_dict, c_dict, 'cache/VQAE2')
    eval_dset = VQAEDataset('val', q_dict, c_dict, 'cache/VQAE2')
    #train_dset = VQAEVQA2Dataset('train', qe_dict, 'cache/VQAE2')
    #eval_dset = VQAEVQA2Dataset('val', qe_dict, 'cache/VQAE2')
    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = utils.factory(constructor, train_dset, args.num_hid, args.att_dim, args.decode_dim).cuda()
    #model = utils.factory(constructor, eval_dset, args.num_hid, args.att_dim, args.decode_dim).cuda()

    if args.resume or args.evaluate:
        model_path = os.path.join(args.output, 'model.pth')
        model_state = torch.load(model_path)
        model.load_state_dict(model_state)
    else:
        #model.w_emb.init_embedding('data/glove6b_init_300d_question.npy')
        #model.w_emb.init_embedding('data/glove6b_init_300d_question_explain.npy')
        #model.w_emb.init_embedding('data/VQAE/glove6b_init_300d_question_explain.npy')
        #model.generator.init_embedding('data/glove6b_init_300d_caption.npy')
        model.w_emb.init_embedding('data/VQAE/glove6b_init_300d_question.npy')
        model.generator.init_embedding('data/VQAE/glove6b_init_300d_explain.npy')
    print('Model has {} parameters in total'.format(utils.params_count(model)))

    if args.parallel:
        model = nn.DataParallel(model).cuda()

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1)
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1)

    if args.evaluate:
        model.train(False)
        eval_score, bound = evaluate(model, eval_loader)
        print('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
    else:
        train(model, train_loader, eval_loader, args.epochs, args.output)
