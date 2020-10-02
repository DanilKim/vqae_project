import os
import time
import json
import torch
import torch.nn as nn
import utils
import pdb
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence


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


def build_optimizer(model):
    optim = torch.optim.Adam([
        {'params': model.w_emb.parameters()},
        {'params': model.q_emb.parameters()},
        {'params': model.v_att.parameters()},
        {'params': model.q_net.parameters()},
        {'params': model.v_net.parameters()},
        {'params': model.classifier.parameters()},
        {'params': model.generator.parameters(), 'lr': 1e-2}
    ], lr=1e-2)
    return optim


def train(model, train_loader, eval_loader, num_epochs, output, dataset):
    #utils.create_dir(output)
    #optim = build_optimizer(model)
    optim = torch.optim.Adamax(model.parameters(), lr=2e-3)
    criterion = nn.CrossEntropyLoss(reduce=(dataset!='VQA2')).cuda()
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0
    print_freq = 10
    A, B, C = evaluate(model, eval_loader, dataset)

    for epoch in range(num_epochs):
        total_vqa_loss = 0
        total_vqe_loss = 0
        train_score = 0
        t = time.time()

        for i, (v, q, a, c, l, f) in enumerate(train_loader):
            bs = v.size(0)
            l, sort_ind = l.sort(dim=0, descending=True)
            ml = l[0]

            v = Variable(v[sort_ind]).cuda()
            q = Variable(q[sort_ind]).cuda()
            a = Variable(a[sort_ind]).cuda()
            c = Variable(c[sort_ind]).cuda()
            if dataset == 'VQA2':
                f = Variable(f[sort_ind], requires_grad=False).cuda().view(bs, 1).float()

            ans_pred, exp_pred = model(v, q, c[:, :-1], f, [j-1 for j in l], ml)
            if False: #i % 100 == 0:
                print('Explain GT:   %s' % (' '.join([train_loader.dataset.explanation_dictionary.idx2word[w.data[0]]
                                                      for id, w in enumerate(c[50]) if id < l[50]])))
                print('Explain Pred: %s' % ('<start> '+' '.join([train_loader.dataset.explanation_dictionary.idx2word[w.data[0]]
                                                      for id, w in enumerate(exp_pred[50].max(1)[1]) if id < (l[50]-1)])))

            if dataset == 'VQA2':
                f = f.expand(bs, ml)
            exp_pred = pack_padded_sequence(exp_pred, [j-1 for j in l], batch_first=True)[0]
            c = pack_padded_sequence(c[:, 1:], [j-1 for j in l], batch_first=True)[0]
            if dataset == 'VQA2':
                f = pack_padded_sequence(f, [j-1 for j in l], batch_first=True)[0]

            vqa_loss = instance_bce_with_logits(ans_pred, a)
            vqe_loss = criterion(exp_pred.cuda(), c)
            if dataset == 'VQA2':
                vqe_loss = sum(vqe_loss * f) / sum(f)
            loss = vqa_loss + vqe_loss
            loss.backward()

            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(ans_pred, a.data).sum()
            total_vqa_loss += vqa_loss.data[0] * v.size(0)
            total_vqe_loss += vqe_loss.data[0] * v.size(0)
            train_score += batch_score

            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'VQE Loss {vqe_loss:.4f}\t'
                  'Batch Time {batch_time:.3f}\t'
                  'VQA Loss {vqa_loss:.4f}\t'
                  'vAcc@1 {acc1:.2f}\t'
                  .format(
                   epoch, i, len(train_loader), vqa_loss=vqa_loss.data[0],
                   vqe_loss=vqe_loss.data[0],
                   acc1=batch_score/v.size(0)*100,
                   batch_time=time.time()-t))
            t = time.time()

        total_vqa_loss /= len(train_loader.dataset)
        total_vqe_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)

        model.train(False)
        gen_score, gt_score, bound = evaluate(model, eval_loader, dataset)
        model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_vqa_loss: %.2f, score: %.2f' % (total_vqa_loss, train_score))
        logger.write('\ttrain_vqe_loss: %.2f' % (total_vqe_loss))
        logger.write('\tgen eval score: %.2f / gt eval score: %.2f (%.2f)' % (100 * gen_score, 100 * gt_score, 100 * bound))

        if gt_score > best_eval_score:
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = gt_score


def evaluate(model, dataloader, dataset):
    score = 0
    gen_score = 0
    gt_score = 0
    upper_bound = 0
    num_data = 0
    results = []
    for i, (v, q, a, c, l, f) in enumerate(dataloader):
        batch_size = v.size(0)

        v = Variable(v, volatile=True).cuda()
        q = Variable(q, volatile=True).cuda()
        c = Variable(c, volatile=True).cuda()
        if dataset == 'VQA2':
            f = Variable(f.float().cuda()).view(batch_size, 1)

        ans_pred_gen, ans_pred_gt, exp_pred = model.evaluate(v, q, f, c, l)

        gen_batch_score = compute_score_with_logits(ans_pred_gen, a.cuda()).sum()
        gt_batch_score = compute_score_with_logits(ans_pred_gt, a.cuda()).sum()
        gen_score += gen_batch_score
        gt_score += gt_batch_score

        upper_bound += (a.max(1)[0]).sum()
        num_data += ans_pred_gen.size(0)

        if i % 100 == 0:
            print('Batch: [%d/%d]\nQuestion: %s\t Answer Prediction: %s\n'
                 'Explain GT:          %s\nExplain Prediction:  %s\n' %
                 (i, len(dataloader),
                  ' '.join([dataloader.dataset.question_dictionary.idx2word[w] for w in
                            q[0].data.tolist() if w > 0]),
                  dataloader.dataset.label2ans[ans_pred_gen[0].max(0)[1].data[0]],
                  ' '.join([dataloader.dataset.explanation_dictionary.idx2word[w] for w in
                            c[0].data.tolist() if w > 2]),
                  ' '.join([dataloader.dataset.explanation_dictionary.idx2word[w] for w in
                            exp_pred[0].data.tolist() if w > 2])))

    gen_score = gen_score / len(dataloader.dataset)
    gt_score = gt_score / len(dataloader.dataset)
    gen_scores = gen_score
    gt_scores = gt_score
    upper_bound = upper_bound / len(dataloader.dataset)
    return gen_score, gt_score, upper_bound
