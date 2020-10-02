import os
import time
import json
import torch
import torch.nn as nn
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils
import pdb
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from PIL import Image, ImageDraw


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


def visualize_attention(split, image_id, question, gt_sentence, pred_sentence, spatial, att_map, save_dir, epoch):
    # L: length of the output sentence
    # sentence: input(train)/output(val) sentence ids (L)
    # spatial: spatial features (location) of regions (36 X 6)
    # att_map: alpha values for regions               (36 X L)
    # split: train / val
    L = len(pred_sentence)
    image_dir = '/data1/coco_raw_images/%s2014' % split
    img_fn = os.path.join(image_dir, 'COCO_%s2014_%012d.jpg' % (split, int(image_id)))
    image = Image.open(img_fn)
    x1 = (image.size[0] * spatial[:,0]).int()
    y1 = (image.size[1] * spatial[:,1]).int()
    x2 = (image.size[0] * (spatial[:,0] + spatial[:,2])).int()
    y2 = (image.size[1] * (spatial[:,1] + spatial[:,3])).int()
    mbox = att_map.max(0)[1]

    fig = plt.figure(figsize=(24.0, 10.0))
    fig.suptitle('Q:   %s?' % question, fontsize=20)
    plt.figtext(0.5, 0.01, 'GT:     %s\nPred:    %s' % (' '.join(gt_sentence), ' '.join(pred_sentence)), horizontalalignment='center', fontsize=18)
    for t in range(L):
        ax = fig.add_subplot(3, 6, t+1)
        ax.set_xticks([])
        ax.set_yticks([])

        img_rgba = image.copy()
        im_a = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(im_a)
        draw.rectangle([(x1[0], y1[0]), (x2[0], y2[0])], fill=int(att_map[0][t]*255))
        img_rgba.putalpha(im_a)
        for i in range(1, int(spatial.size(0))):
            img_rgb = image.copy()
            im_a = Image.new("L", image.size, 0)
            draw = ImageDraw.Draw(im_a)
            draw.rectangle([(x1[i], y1[i]), (x2[i], y2[i])], fill=int(att_map[i][t]*255))
            img_rgb.putalpha(im_a)
            img_rgba = Image.alpha_composite(img_rgba, img_rgb)

        ax.imshow(img_rgba)
        m = int(mbox[t])
        ax.add_patch(
                plt.Rectangle((x1[m], y1[m]), x2[m]-x1[m], y2[m]-y1[m], fill=False, edgecolor='red', linewidth=1)
        )
        ax.text(x1[m], y1[m]-11, pred_sentence[t], bbox=dict(facecolor='blue', alpha=0.5), fontsize=8, color='white')

    plt.savefig(os.path.join(save_dir, 'COCO_%s2014_%012d_ep_%d.jpg' % (split, int(image_id), epoch)))
    plt.clf()


def train(model, train_loader, eval_loader, num_epochs, output, model_fn):
    #utils.create_dir(output)
    #optim = build_optimizer(model)
    optim = torch.optim.Adamax(model.parameters(), lr=2e-3)
    criterion = nn.CrossEntropyLoss(reduce=True).cuda()
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0
    print_freq = 10
    image_dir = '/data1/coco_raw_images/train2014'
    #evaluate(model, eval_loader)

    idx2word = train_loader.dataset.explanation_dictionary.idx2word
    for epoch in range(num_epochs):
        total_vqa_loss = 0
        total_vqe_loss = 0
        train_score = 0
        t = time.time()
        for i, (v, s, iid, q, a, c, l) in enumerate(train_loader):
            bs = v.size(0)
            l, sort_ind = l.sort(dim=0, descending=True)
            ml = l[0]

            iid = iid[sort_ind]
            s = s[sort_ind]
            v = Variable(v[sort_ind]).cuda()
            q = Variable(q[sort_ind]).cuda()
            a = Variable(a[sort_ind]).cuda()
            c = Variable(c[sort_ind]).cuda()

            exp_pred, alphas = model(v, q, c[:, :-1], [j-1 for j in l], ml)
            #if i % 100 == 0:
            if epoch > 10 and i % 100 == 0:
                q_sent = ' '.join([train_loader.dataset.question_dictionary.idx2word[w.data[0]] for w in q[50] if w.data[0] > 0])
                gt_tokens = [idx2word[w.data[0]] for id, w in enumerate(c[50]) if id < l[50]]
                pred_tokens = [idx2word[w.data[0]] for id, w in enumerate(exp_pred[50].max(1)[1]) if id < (l[50]-1)]
                print('Question:   %s?' % q_sent)
                print('Explain GT:   %s' % (' '.join(gt_tokens)))
                print('Explain Pred: %s' % ('<start> '+' '.join(pred_tokens)))
                #visualize_attention('train', iid[50], q_sent, gt_tokens, pred_tokens, s[50], alphas[50], output, epoch)

            exp_pred = pack_padded_sequence(exp_pred, [j-1 for j in l], batch_first=True)[0]
            c = pack_padded_sequence(c[:, 1:], [j-1 for j in l], batch_first=True)[0]

            vqe_loss = criterion(exp_pred.cuda(), c)
            vqe_loss.backward()

            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            total_vqe_loss += vqe_loss.data[0] * v.size(0)

            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'VQE Loss {vqe_loss:.4f}\t'
                  'Batch Time {batch_time:.3f}\t'
                  .format(
                   epoch, i, len(train_loader),
                   vqe_loss=vqe_loss.data[0],
                   batch_time=time.time()-t))
            t = time.time()

        total_vqe_loss /= len(train_loader.dataset)

        if epoch % 2 == 0:
            model.train(False)
            evaluate(model, eval_loader, output)
            #eval_score, bound = evaluate(model, eval_loader)
            model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_vqe_loss: %.2f' % (total_vqe_loss))
        #logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        #if eval_score > best_eval_score:
        model_path = os.path.join(output, model_fn)
        torch.save(model.state_dict(), model_path)
        #best_eval_score = eval_score


def evaluate(model, dataloader, output):
    score = 0
    upper_bound = 0
    num_data = 0
    results = []

    idx2word = dataloader.dataset.explanation_dictionary.idx2word
    for i, (v, s, iid, q, a, c, l) in enumerate(dataloader):
        batch_size = v.size(0)

        v = Variable(v, volatile=True).cuda()
        q = Variable(q, volatile=True).cuda()
        exp_pred, alphas, _ = model.evaluate(v, q)

        if i % 100 == 0:
            q_sent = ' '.join([dataloader.dataset.question_dictionary.idx2word[w] for w in q[0].data.tolist() if w > 0])
            gt_tokens = [idx2word[w] for w in c[0].tolist() if w > 2]
            pred_tokens = [idx2word[w] for w in exp_pred[0].data.tolist() if w > 2]
            print('Batch: [%d/%d]\nQuestion: %s\n'
                 'Explain GT:          %s\nExplain Prediction:  %s\n' %
                 (i, len(dataloader),
                  q_sent,
                  ' '.join(gt_tokens),
                  ' '.join(pred_tokens)))
            #visualize_attention('val', iid[0], q_sent, gt_tokens, pred_tokens, s[0], alphas[0], output, -1)
