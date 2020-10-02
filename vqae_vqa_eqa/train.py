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


def train(model, train_loader, eval_loader, num_epochs, output, temperature=1.0):
    #utils.create_dir(output)
    #optim = build_optimizer(model)
    optim = torch.optim.Adamax(model.parameters(), lr=2e-3)
    criterion = nn.CrossEntropyLoss(reduce=True).cuda()
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0
    print_freq = 10
    A, B = evaluate(model, eval_loader)

    for epoch in range(num_epochs):
        total_vqa_loss = 0
        total_vqe_loss = 0
        vq_score = 0
        eq_score = 0
        train_score = 0
        t = time.time()

        for i, (v, q, a, c, l) in enumerate(train_loader):
            bs = v.size(0)
            l, sort_ind = l.sort(dim=0, descending=True)
            ml = l[0]

            v = Variable(v[sort_ind]).cuda()
            q = Variable(q[sort_ind]).cuda()
            a = Variable(a[sort_ind]).cuda()
            c = Variable(c[sort_ind]).cuda()

            ans_pred, exp_pred = model(v, q, c[:, :-1], [j-1 for j in l], ml) #, temperature)
            if False: #i % 100 == 0:
                print('Explain GT:   %s' % (' '.join([train_loader.dataset.explanation_dictionary.idx2word[w.data[0]]
                                                      for id, w in enumerate(c[50]) if id < l[50]])))
                print('Explain Pred: %s' % ('<start> '+' '.join([train_loader.dataset.explanation_dictionary.idx2word[w.data[0]]
                                                      for id, w in enumerate(exp_pred[50].max(1)[1]) if id < (l[50]-1)])))

            #f = Variable(f.float().cuda()).view(bs, 1).expand(bs, ml)
            exp_pred = pack_padded_sequence(exp_pred, [j-1 for j in l], batch_first=True)[0]
            c = pack_padded_sequence(c[:, 1:], [j-1 for j in l], batch_first=True)[0]
            #f = pack_padded_sequence(f, [j-1 for j in l], batch_first=True)[0]

            if type(ans_pred) is list:
                vqa_loss = instance_bce_with_logits(ans_pred[0], a) + instance_bce_with_logits(ans_pred[1], a)
                vq_batch_score = compute_score_with_logits(ans_pred[0], a.data).sum()
                eq_batch_score = compute_score_with_logits(ans_pred[1], a.data).sum()
                vq_score += vq_batch_score
                eq_score += eq_batch_score
                ans_pred = ans_pred[0] + ans_pred[1]
            else:
                vqa_loss = instance_bce_with_logits(ans_pred, a)
            vqe_loss = criterion(exp_pred.cuda(), c)
            #vqe_loss = sum(vqe_loss * f) / sum(f)
            loss = vqe_loss #vqa_loss + vqe_loss
            loss.backward()

            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            #pdb.set_trace()
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
                  #'VQ_Acc@1 {vq_acc1:.2f}\t'
                  #'EQ_Acc@1 {eq_acc1:.2f}\t'
                  'vAcc@1 {acc1:.2f}\t'
                  .format(
                   epoch, i, len(train_loader), vqa_loss=vqa_loss.data[0],
                   vqe_loss=vqe_loss.data[0],
                   #vq_acc1=vq_batch_score/v.size(0)*100,
                   #eq_acc1=eq_batch_score/v.size(0)*100,
                   acc1=batch_score/v.size(0)*100,
                   batch_time=time.time()-t))
            t = time.time()

        total_vqa_loss /= len(train_loader.dataset)
        total_vqe_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)

        model.train(False)
        eval_score, bound = evaluate(model, eval_loader)
        model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_vqa_loss: %.2f, score: %.2f' % (total_vqa_loss, train_score))
        logger.write('\ttrain_vqe_loss: %.2f' % (total_vqe_loss))
        if type(eval_score) is list:
            logger.write('\teval score: vqa = %.2f / eqa = %.2f  / ens = %.2f (%.2f)'
                         % (100 * eval_score[0], 100 * eval_score[1], 100 * eval_score[2], 100 * bound))
            eval_score = eval_score[2]
        else:
            logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        if True: #eval_score > best_eval_score:
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score


def evaluate(model, dataloader, temperature=1.0):
    score = 0
    vq_score = 0
    eq_score = 0
    upper_bound = 0
    num_data = 0
    results = []
    for i, (v, q, a, c, l) in enumerate(dataloader):
        batch_size = v.size(0)

        v = Variable(v, volatile=True).cuda()
        q = Variable(q, volatile=True).cuda()
        c = Variable(c, volatile=True).cuda()
        #ans_pred, exp_pred = model.evaluate(v, q, c, temperature)
        ans_pred, exp_pred = model.evaluate(v, q)

        if type(ans_pred) is list:
            vq_batch_score = compute_score_with_logits(ans_pred[0], a.cuda()).sum()
            eq_batch_score = compute_score_with_logits(ans_pred[1], a.cuda()).sum()
            vq_score += vq_batch_score
            eq_score += eq_batch_score
            ans_pred = ans_pred[0] + ans_pred[1]

        batch_score = compute_score_with_logits(ans_pred, a.cuda()).sum()
        score += batch_score

        upper_bound += (a.max(1)[0]).sum()
        num_data += ans_pred[0].size(0)
        #num_data += ans_pred.size(0)

        if i % 100 == 0:
            print('Batch: [%d/%d]\nQuestion: %s\t Answer Prediction: %s\n'
                 'Explain GT:          %s\nExplain Prediction:  %s\n' %
                 (i, len(dataloader),
                  ' '.join([dataloader.dataset.question_dictionary.idx2word[w] for w in
                            q[0].data.tolist() if w > 0]),
                  dataloader.dataset.label2ans[ans_pred[0].max(0)[1].data[0]],
                  ' '.join([dataloader.dataset.explanation_dictionary.idx2word[w] for w in
                            c[0].data.tolist() if w > 2]),
                  ' '.join([dataloader.dataset.explanation_dictionary.idx2word[w] for w in
                            exp_pred[0].data.tolist() if w > 2])))

    score = score / len(dataloader.dataset)

    if vq_score == 0:
        scores = score
    else:
        vq_score = vq_score / len(dataloader.dataset)
        eq_score = eq_score / len(dataloader.dataset)
        scores = [vq_score, eq_score, score]
    upper_bound = upper_bound / len(dataloader.dataset)
    return scores, upper_bound
