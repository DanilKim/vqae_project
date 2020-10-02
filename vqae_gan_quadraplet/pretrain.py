import argparse
import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
import utils
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

from dataset import Dictionary, VQASemiDataset, VQAEDataset, VQAFullDataset, data_factory
import base_model
import pdb


def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def pretrain_ans(model, train_loader, eval_loader, num_epoch, output):
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'pretrain_VQA_log.txt'))

    best_score = 0
    model.train(True)
    for epoch in range(num_epoch):
        total_vqa_loss = 0
        train_score = 0
        print_freq = 10
        t = time.time()

        for i, (v, q, a) in enumerate(train_loader):
            v = Variable(v).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()

            ans_pred = model(v, q)

            loss = instance_bce_with_logits(ans_pred, a)
            loss.backward()

            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(ans_pred, a.data).sum()
            total_vqa_loss += loss.data[0] * v.size(0)
            train_score += batch_score

            if i % print_freq == 0:
                print('Pretrain A Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time:.3f}\t'
                  'VQA Loss {vqa_loss:.4f}\t'
                  'Acc {acc:.2f}\t'
                  .format(
                   epoch, i, len(train_loader), vqa_loss=loss.data[0],
                   acc=batch_score/v.size(0)*100, batch_time=time.time()-t))
            t = time.time()

        total_vqa_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)

        model.train(False)
        eval_score, upper_bound = eval_ans(model, eval_loader)
        model.train(True)
        logger.write('A epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, train_score: %.2f' % (total_vqa_loss, train_score))
        logger.write('\teval_score: %.2f (%.2f)' % (100 * eval_score, 100 * upper_bound))

        if best_score < eval_score:
            model_path = os.path.join(output, 'pretrained_VQA.pth')
            torch.save(model.state_dict(), model_path)


def eval_ans(model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0
    results = []
    t = time.time()
    for i, (v, q, a) in enumerate(dataloader):
        batch_size = v.size(0)

        v = Variable(v, volatile=True).cuda()
        q = Variable(q, volatile=True).cuda()

        ans_pred = model(v, q)
        batch_score = compute_score_with_logits(ans_pred, a.cuda()).sum()
        score += batch_score

        upper_bound += (a.max(1)[0]).sum()
        num_data += ans_pred.size(0)

        if i % 100 == 0:
            print('Batch: [%d/%d]\nQuestion: %s\t Answer Prediction: %s\n' %
                 (i, len(dataloader),
                  ' '.join([dataloader.dataset.question_dictionary.idx2word[w] for w in
                            q[0].data.tolist() if w > 0]),
                  dataloader.dataset.label2ans[ans_pred[0].max(0)[1].data[0]]))

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    return score, upper_bound


def pretrain_gen(model, dataloader, num_epoch, output, dataset):
    optim = torch.optim.Adamax(model.generator.parameters())
    gCrit = nn.CrossEntropyLoss(reduce=True).cuda()
    logger = utils.Logger(os.path.join(output, 'pretrain_VQE_log.txt'))

    model.train(True)
    for epoch in range(num_epoch):
        total_vqe_loss = 0
        print_freq = 10
        t = time.time()

        for i, (v, q, a, c, l) in enumerate(dataloader):
            bs = v.size(0)
            l, sort_ind = l.sort(dim=0, descending=True)
            ml = l[0]

            v = Variable(v[sort_ind]).cuda()
            q = Variable(q[sort_ind]).cuda()
            a = Variable(a[sort_ind]).cuda()
            c = Variable(c[sort_ind]).cuda()

            exp_pred = model.generate(v, q, c[:, :-1], [j-1 for j in l], ml, 1.0, True)

            if i % 100 == 0:
                print('Explain GT:   %s' % (' '.join([dataloader.dataset.explanation_dictionary.idx2word[w.data[0]]
                                                      for id, w in enumerate(c[50]) if id < l[50]])))
                print('Explain Pred: %s' % ('<start> '+' '.join([dataloader.dataset.explanation_dictionary.idx2word[w.data[0]]
                                                      for id, w in enumerate(exp_pred[50].max(1)[1]) if id < (l[50]-1)])))

            exp_pred = pack_padded_sequence(exp_pred, [j-1 for j in l], batch_first=True)[0]
            c = pack_padded_sequence(c[:, 1:], [j-1 for j in l], batch_first=True)[0]

            loss = gCrit(exp_pred.cuda(), c)
            loss.backward()

            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            total_vqe_loss += loss.data[0] * v.size(0)

            if i % print_freq == 0:
                print('Pretrain G Epoch: [{0}][{1}/{2}]\t'
                  'VQE Loss {vqe_loss:.4f}\t'
                  'Batch Time {batch_time:.3f}\t'
                  .format(
                   epoch, i, len(dataloader),
                   vqe_loss=loss.data[0],
                   batch_time=time.time()-t))
            t = time.time()

        total_vqe_loss /= len(dataloader.dataset)
        logger.write('A epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_vqe_loss: %.2f' % total_vqe_loss)

    model_path = os.path.join(output, 'pretrained_VQE.pth')
    generator_dict = {k: v for k, v in model.state_dict().items() if k.startswith('generator')}
    torch.save(generator_dict, model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--att_dim', type=int, default=512)
    parser.add_argument('--decode_dim', type=int, default=1024)
    parser.add_argument('--model', type=str, default='newatt_2')
    parser.add_argument('--share_qe_dict', type=bool, default=False)
    parser.add_argument('--train_set', type=str, default='VQA2', choices=['VQAE', 'VQA2'])
    parser.add_argument('--val_set', type=str, default='VQA2', choices=['VQAE', 'VQA2'])
    parser.add_argument('--output', type=str, default='saved_models/semi_pA15_pE20_GAN_VQAE')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--pretrain', type=str, default='VQA', choices=['VQA', 'VQE'])
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--evaluate', type=str, default=None)
    parser.add_argument('--parallel', type=bool, default=False)
    args = parser.parse_args()

    vocab_source = 'VQA2' #args.train_set[-4:]
    target_source = vocab_source + '_2_' + args.val_set

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

    if args.evaluate is not None:
        args.train_set = args.val_set

    train_dset = data_factory(args.train_set, 'train', q_dict, c_dict, os.path.join('cache', target_source))
    eval_dset = data_factory(args.val_set, 'val', q_dict, c_dict, os.path.join('cache', target_source))
    batch_size = args.batch_size

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1)
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1)

    constructor = 'build_%s_%s' % (args.pretrain, args.model)
    model = utils.model_factory(constructor, train_dset, args.num_hid, args.att_dim, args.decode_dim).cuda()

    if args.resume or (args.evaluate is not None):
        model_path = os.path.join(args.output, args.evaluate)
        model_state = torch.load(model_path)
        model.load_state_dict(model_state)
    elif args.pretrained is not None:
        pretrained_path = os.path.join(args.output, args.pretrained)
        pretrained_dict = torch.load(pretrained_path)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        print('From Scratch...')
        model.w_emb.init_embedding(q_emb_file)
        if args.pretrain == 'VQE':
            model.generator.init_embedding(c_emb_file)

    print('Model has {} parameters in total'.format(utils.params_count(model)))

    if args.parallel:
        model = nn.DataParallel(model).cuda()

    if args.evaluate is not None:
        model.train(False)
        eval_score, bound = eval_ans(model, eval_loader)
        print('\teval score: vqa = %.2f' % (100 * eval_score))
    elif args.pretrain == 'VQA':
        pretrain_ans(model, train_loader, eval_loader, args.epochs, args.output)
    elif args.pretrain == 'VQE':
        pretrain_gen(model, train_loader, args.epochs, args.output, args.train_set)
    else:
        ValueError
