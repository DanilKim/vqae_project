import os
import time
import json
import torch
import torch.nn as nn
import utils
import pdb
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from discriminator import dTrain, dAdvTrain, dEval, RFdiscriminator, VQAEdiscriminator


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


def compute_bc_accuracy(logits, labels):
    pred = logits >= 0.5
    num_correct = (pred == labels.byte().cuda()).sum()
    acc = float(num_correct) / len(labels)
    return acc


def build_gen_optimizer(model):
    optim = torch.optim.Adam([
        {'params': model.generator.parameters()},
        {'params': model.e_emb.rnn.parameters()},
        {'params': model.e_net.parameters()},
        {'params': model.classifier.parameters()},
    ], lr=2e-3)
    return optim


def build_disc_optimizer(model, discs, lrs):
    params = [
        {'params': model.e_emb.rnn.parameters()},
        {'params': model.e_net.parameters()},
        {'params': model.classifier.parameters()}
    ]
    for (disc, lr) in zip(discs, lrs):
        params.append({'params': disc.parameters(), 'lr': lr})
    optim = torch.optim.Adamax(params, lr=2e-3)
    return optim


def build_cls_optimizer(model):
    optim = torch.optim.Adam([
        {'params': model.e_emb.parameters(), 'lr': 2e-3},
        {'params': model.classifier.parameters(), 'lr': 2e-3},
    ], lr=2e-3)
    return optim


def train(model, train_loader, eval_loader, num_epochs, output):
    optim = torch.optim.Adamax(model.parameters(), lr=2e-3)
    gCrit = nn.CrossEntropyLoss(reduce=True).cuda()

    proj_dim = 512
    RFD = RFdiscriminator(model.generator.decode_dim).cuda()
    VQAED = VQAEdiscriminator(model.generator.vis_dim, model.q_emb.num_hid, model.w_emb.emb_dim,
                              model.generator.decode_dim, proj_dim).cuda()
    #disc = BCdiscriminator(model.generator.decode_dim).cuda()
    #dCrit = nn.BCELoss().cuda()

    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0
    pde = 5
    pge = 20
    #A, B = evaluate(model, eval_loader)

    ################  Pre-train Generator 25 epochs  ##################
    #model.train(True)
    #g_train_score = train_gen(model, RFD, VED, QED, AED, gCrit, train_loader, 23, pge, adv=0)
    #g_train_score = 0
    #model.train(False)
    #evaluate(model, eval_loader, pge, g_train_score, logger)
    ###################################################################

    ################  Pre-train Discriminator 10 epochs  ##############
    #model.train(True); RFD.train(True); VED.train(True); QED.train(True); AED.train(True)
    #train_score, train_loss = train_disc(model, RFD, VED, QED, AED, train_loader, 1, pde)
    #model.train(False); RFD.train(False); VED.train(False); QED.train(False); AED.train(False)
    #eval_disc(model, RFD, VED, QED, eval_loader, pde, train_score, train_loss, logger)
    ###################################################################

    #####################  Iterative D & G train  #####################
    d = 1
    g = 1
    model.train(False)
    g_eval_score = evaluate(model, eval_loader, 0, [0.0], logger)
    model.train(True)
    for epoch in range(1, num_epochs+1, 1):
        ################  Train D  #################
        model.train(True); RFD.train(True); VQAED.train(True);
        d_train_score, d_train_loss = train_disc(model, RFD, VQAED, train_loader, d*epoch, d*(epoch+1))
        model.train(False); RFD.train(False); VQAED.train(False);
        d_eval_score = eval_disc(model, RFD, VQAED, eval_loader, d*(epoch+1)-1, d_train_score, d_train_loss, logger)

        ################  Train G  #################
        if d_eval_score[0] > 75:
            model.train(True)
            g_train_score = train_gen(model, RFD, VQAED, gCrit, train_loader, g*epoch, g*(epoch+1))
            model.train(False); RFD.train(False); VQAED.train(False);
            g_eval_score = evaluate(model, eval_loader, g*(epoch+1)-1, g_train_score, logger)

        if g_eval_score > best_eval_score:
            model_path = os.path.join(output, 'best_model.pth')
            RFD_path = os.path.join(output, 'best_RFD.pth')
            VQAED_path = os.path.join(output, 'best_VQAED.pth')
            torch.save(model.state_dict(), model_path)
            torch.save(RFD.state_dict(), RFD_path)
            torch.save(VQAED.state_dict(), VQAED_path)
            best_eval_score = g_eval_score

        model_path = os.path.join(output, 'model.pth')
        RFD_path = os.path.join(output, 'RFD.pth')
        VQAED_path = os.path.join(output, 'VQAED.pth')
        torch.save(model.state_dict(), model_path)
        torch.save(RFD.state_dict(), RFD_path)
        torch.save(VQAED.state_dict(), VQAED_path)
    ##################################################################

def train_disc(model, RFD, VQAED, dataloader, start_epoch, end_epoch):
    disc_optim = build_disc_optimizer(model, [RFD, VQAED], [2e-3, 2e-3])
    for epoch in range(start_epoch, end_epoch, 1):
        data_count = 0
        rf_count = 0
        total_vqa_loss = 0
        total_vqa_score = 0
        total_rf_loss, total_se_loss = (0, 0)
        total_rf_score, total_se_score = (0, 0)
        print_freq = 10
        t = time.time()

        for i, (L, UL) in enumerate(dataloader):
            bs = L['features'].size(0)
            assert bs > 2
            l, sort_ind = L['exp_len'].sort(dim=0, descending=True)
            ml = l[0]

            L_v = Variable(L['features'][sort_ind]).cuda()
            L_q = Variable(L['question'][sort_ind]).cuda()
            L_e = Variable(L['explain'][sort_ind]).cuda()
            L_a = Variable(L['target'][sort_ind]).cuda()

            U_v = Variable(UL['features'][sort_ind]).cuda()
            U_q = Variable(UL['question'][sort_ind]).cuda()
            U_a = Variable(UL['target'][sort_ind]).cuda()

            ans_pred, exp_pred, L_real_q, L_real_hidden, L_fake_hidden \
                = model(L_v, L_q, L_e[:, :-1], [j-1 for j in l], ml)
            L_gt_hidden = model.dec_enc(None, L_e)
            U_exp_pred, _, U_real_q, U_fake_hidden = model.evaluate(U_v, U_q)
            L_fake_v = torch.cat([L_v[1:], L_v[0].unsqueeze(0)], 0)
            L_fake_q = torch.cat([L_real_q[1:], L_real_q[0].unsqueeze(0)], 0)
            L_fake_a = torch.cat([L_a[1:], L_a[0].unsqueeze(0)], 0)
            U_fake_v = torch.cat([U_v[1:], U_v[0].unsqueeze(0)], 0)
            U_fake_q = torch.cat([U_real_q[1:], U_real_q[0].unsqueeze(0)], 0)
            U_fake_a = torch.cat([U_a[1:], U_a[0].unsqueeze(0)], 0)

            #######################    Loss & Score on Labeled Data    ########################
            vqa_loss = instance_bce_with_logits(ans_pred, L_a)
            r_pred, f_pred, r_label, f_label, rf_loss = dTrain(RFD, L_real_hidden, L_fake_hidden)
            ser_pred, sef_pred, ser_label, sef_label, se_loss \
                = dTrain(VQAED, [L_v, L_real_q, L_a, L_real_hidden], [L_fake_v, L_fake_q, L_fake_a, L_fake_hidden])

            loss = vqa_loss + (rf_loss + se_loss)

            VQA_score = compute_score_with_logits(ans_pred, L_a.data).sum()
            RFD_score = compute_bc_accuracy(torch.cat([r_pred, f_pred], 0).data, torch.cat([r_label, f_label], 0).data)
            SED_score = compute_bc_accuracy(torch.cat([ser_pred, sef_pred], 0).data, torch.cat([ser_label, sef_label], 0).data)

            total_vqa_loss += vqa_loss.data[0] * L_v.size(0)
            total_rf_loss += rf_loss.data[0] * L_v.size(0)
            total_se_loss += se_loss.data[0] * L_v.size(0)
            total_vqa_score += VQA_score
            total_rf_score += RFD_score * L_v.size(0)
            total_se_score += SED_score * L_v.size(0)
            data_count += L_v.size(0)
            rf_count += L_v.size(0)

            if i % print_freq == 0:
                print('Train D Epoch: [{0}][{1}/{2}] - Batch Time {batch_time:.3f} \n\tLabeled Losses   - '
                  'VQA Loss {vqa_loss:.3f} / SED Loss {se_loss:.3f} / RFD Loss {rf_loss:.3f}'
                  '\n\tLabeled Scores   - '
                  'VQA Acc {vqa_acc:.2f}  / SED Acc {se_acc: .2f}  / RFD Acc {rf_acc: .2f}'
                  .format(
                   epoch, i, len(dataloader), batch_time=time.time()-t,
                   vqa_loss=vqa_loss.data[0], vqa_acc=VQA_score/(L_v.size(0)) * 100,
                   se_loss=se_loss.data[0],   se_acc=SED_score * 100,
                   rf_loss=rf_loss.data[0],   rf_acc=RFD_score * 100,
               ))

            #######################    Loss & Score on Unlabeled Data    ########################
            vqa_loss = instance_bce_with_logits(ans_pred, U_a)
            ser_pred, sef_pred, ser_label, sef_label, se_loss \
                = dTrain(VQAED, [U_v, U_real_q, U_a, U_fake_hidden], [U_fake_v, U_fake_q, U_fake_a, U_fake_hidden])

            loss = loss + vqa_loss + se_loss

            VQA_score = compute_score_with_logits(ans_pred, U_a.data).sum()
            SED_score = compute_bc_accuracy(torch.cat([ser_pred, sef_pred], 0).data, torch.cat([ser_label, sef_label], 0).data)

            total_vqa_loss += vqa_loss.data[0] * U_v.size(0)
            total_se_loss += se_loss.data[0] * U_v.size(0)
            total_vqa_score += VQA_score
            total_se_score += SED_score * U_v.size(0)
            data_count += U_v.size(0)

            if i % print_freq == 0:
                print('\tUnlabeled Losses - '
                  'VQA Loss {vqa_loss:.3f} / SED Loss {se_loss:.3f}'
                  '\n\tUnlabeled Scores - '
                  'VQA Acc {vqa_acc:.2f}  / SED Acc {se_acc: .2f}'
                  .format(
                   vqa_loss=vqa_loss.data[0], vqa_acc=VQA_score/(U_v.size(0))*100,
                   se_loss=se_loss.data[0],   se_acc=SED_score * 100,
                   ))
            t = time.time()

            #######################    Optimization Step    ###########################
            loss.backward()
            #nn.utils.clip_grad_norm(disc.parameters(), 0.25)
            disc_optim.step()
            disc_optim.zero_grad()

        total_vqa_loss = total_vqa_loss / data_count
        total_rf_loss = total_rf_loss / rf_count
        total_se_loss = total_se_loss / data_count
        total_vqa_score = 100 * total_vqa_score / data_count
        total_rf_score  = 100 * total_rf_score  / rf_count
        total_se_score  = 100 * total_se_score  / data_count

        dataloader.dataset.shuffled_unlabeled = torch.randperm(dataloader.dataset.unlabeled_cnt)
        scores = [total_vqa_score, total_rf_score, total_se_score]
        losses = [total_vqa_loss, total_rf_loss, total_se_loss]
    return scores, losses

def train_gen(model, RFD, VQAED, gCrit, dataloader, start_epoch, end_epoch, adv=1):
    gen_optim = build_gen_optimizer(model)
    for epoch in range(start_epoch, end_epoch, 1):
        data_count = 0
        labeled_cnt = 0
        total_vqa_loss = 0
        total_vqe_loss = 0
        total_vqa_score = 0
        print_freq = 10
        t = time.time()

        for i, (L, UL) in enumerate(dataloader):
            bs = L['features'].size(0)
            l, sort_ind = L['exp_len'].sort(dim=0, descending=True)
            ml = l[0]

            L_v = Variable(L['features'][sort_ind]).cuda()
            L_q = Variable(L['question'][sort_ind]).cuda()
            L_a = Variable(L['target'][sort_ind]).cuda()
            L_e = Variable(L['explain'][sort_ind]).cuda()

            U_v = Variable(UL['features'][sort_ind]).cuda()
            U_q = Variable(UL['question'][sort_ind]).cuda()
            U_a = Variable(UL['target'][sort_ind]).cuda()

            ans_pred, exp_pred, L_real_q, L_real_hidden, L_fake_hidden \
                = model(L_v, L_q, L_e[:, :-1], [j-1 for j in l], ml)
            U_exp_pred, _, U_real_q, U_fake_hidden = model.evaluate(U_v, U_q)

            if i % 100 == 0:
                print('Explain GT:   %s' % (' '.join([dataloader.dataset.explanation_dictionary.idx2word[w.data[0]]
                                                      for id, w in enumerate(L_e[50]) if id < l[50]])))
                print('Explain Pred: %s' % ('<start> '+' '.join([dataloader.dataset.explanation_dictionary.idx2word[w.data[0]]
                                                      for id, w in enumerate(exp_pred[50].max(1)[1]) if id < (l[50]-1)])))

            exp_pred = pack_padded_sequence(exp_pred, [j-1 for j in l], batch_first=True)[0]
            L_e = pack_padded_sequence(L_e[:, 1:], [j-1 for j in l], batch_first=True)[0]

            #############  Loss & Score on Labeled Data  #############
            vqa_loss = instance_bce_with_logits(ans_pred, L_a)
            vqe_loss = gCrit(exp_pred.cuda(), L_e)
            rf_loss = dAdvTrain(RFD, L_fake_hidden)
            se_loss = dAdvTrain(VQAED, [L_v, L_real_q, L_a, L_fake_hidden])
            d_loss = rf_loss + se_loss

            loss = vqa_loss + (adv * d_loss) #+ vqe_loss

            VQA_score = compute_score_with_logits(ans_pred, L_a.data).sum()
            total_vqa_loss += vqa_loss.data[0] * L_v.size(0)
            total_vqe_loss += vqe_loss.data[0] * L_v.size(0)
            total_vqa_score += VQA_score
            data_count += L_v.size(0)
            labeled_cnt += L_v.size(0)

            if i % print_freq == 0:
                print('Train G Epoch: [{0}][{1}/{2}] - Batch Time {batch_time:.3f} \n\tLabeled -   '
                  'VQE Loss {vqe_loss:.4f}\t'
                  'VQA Loss {vqa_loss:.4f}\t'
                  'Acc {acc:.2f}\t'
                  .format(
                   epoch, i, len(dataloader), batch_time=time.time()-t, vqa_loss=vqa_loss.data[0],
                   vqe_loss=vqe_loss.data[0], acc=VQA_score/(L_v.size(0))*100
                ))

            #############  Loss & Score on UnLabeled Data  #############
            vqa_loss = instance_bce_with_logits(ans_pred, L_a)
            rf_loss = dAdvTrain(RFD, U_fake_hidden)
            se_loss = dAdvTrain(VQAED, [U_v, U_real_q, U_a, U_fake_hidden])
            d_loss = rf_loss + se_loss

            loss = loss + vqa_loss + (adv * d_loss)

            VQA_score = compute_score_with_logits(ans_pred, U_a.data).sum()
            data_count += U_v.size(0)
            total_vqa_loss += vqa_loss.data[0] * U_v.size(0)
            total_vqa_score += VQA_score
            data_count += U_v.size(0)

            ##################  Optimization Step  ###################
            loss.backward()

            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            gen_optim.step()
            gen_optim.zero_grad()

            if i % print_freq == 0:
                print('\tUnlabeled - '
                  'VQA Loss {vqa_loss:.4f}\t'
                  'Acc {acc:.2f}\t'
                  .format(
                   vqa_loss=vqa_loss.data[0],
                   acc=VQA_score/(U_v.size(0))*100,
                  ))
            t = time.time()

        total_vqa_loss /= data_count
        total_vqe_loss /= labeled_cnt
        total_vqa_score = 100 * total_vqa_score / data_count
        return total_vqa_score, [total_vqa_loss, total_vqe_loss]

def eval_disc(model, RFD, VQAED, dataloader, epoch, train_scores, train_losses, logger):
    total_vqa_score = 0
    total_rf_score, total_se_score= (0, 0)
    t = time.time()

    for i, E in enumerate(dataloader):
        l, sort_ind = E['exp_len'].sort(dim=0, descending=True)
        L = (l > 0).sum()
        ml = l[0]

        v = Variable(E['features'][sort_ind]).cuda()
        q = Variable(E['question'][sort_ind]).cuda()
        c = Variable(E['explain'][sort_ind]).cuda()
        a = Variable(E['target'][sort_ind]).cuda()

        _, vqe_pred = model.generate(v[:L], q[:L], c[:L, :-1], [j-1 for j in l[:L] if j > 0], ml, 1.0, True)
        real_hidden = model.dec_enc(None, vqe_pred)
        ans_pred, _, real_q, fake_hidden = model.evaluate(v, q)
        fake_v = torch.cat([v[1:], v[0].unsqueeze(0)], 0)
        fake_q = torch.cat([real_q[1:], real_q[0].unsqueeze(0)], 0)
        fake_a = torch.cat([a[1:], a[0].unsqueeze(0)], 0)

        r_pred, f_pred, r_label, f_label = dEval(RFD, real_hidden, fake_hidden)
        ser_pred, sef_pred, ser_label, sef_label = dEval(VQAED, [v, real_q, a, fake_hidden], [fake_v, fake_q, fake_a, fake_hidden])

        VQA_score = compute_score_with_logits(ans_pred, a.data).sum()
        RFD_score = compute_bc_accuracy(torch.cat([r_pred, f_pred], 0).data, torch.cat([r_label, f_label], 0).data)
        SED_score = compute_bc_accuracy(torch.cat([ser_pred, sef_pred], 0).data, torch.cat([ser_label, sef_label], 0).data)

        total_vqa_score += VQA_score
        total_rf_score += RFD_score * v.size(0)
        total_se_score += SED_score * v.size(0)

    total_vqa_score = 100 * total_vqa_score / len(dataloader.dataset)
    total_rf_score = 100 * total_rf_score / len(dataloader.dataset)
    total_se_score = 100 * total_se_score / len(dataloader.dataset)

    logger.write('D epoch %d, time: %.2f' % (epoch, time.time()-t))
    logger.write('\ttrain vqa score %.2f' % train_scores[0])
    logger.write('\teval vqa score %.2f' % total_vqa_score)
    logger.write('\ttrain disc loss: RF - %.4f / SE - %.4f' % (train_losses[1], train_losses[2]))
    logger.write('\ttrain disc score: RF - %.2f / SE - %.2f' % (train_scores[1], train_scores[2]))
    logger.write('\teval disc score: RF - %.2f / SE - %.2f' % (total_rf_score, total_se_score))
    return [total_rf_score, total_se_score]

def evaluate(model, dataloader, epoch, train_score, logger):
    score = 0
    upper_bound = 0
    num_data = 0
    results = []
    t = time.time()
    for i, E in enumerate(dataloader):
        batch_size = E['features'].size(0)

        v = Variable(E['features'], volatile=True).cuda()
        q = Variable(E['question'], volatile=True).cuda()
        a = Variable(E['target'], volatile=True).cuda()
        c = E['explain'].cuda()

        ans_pred, exp_pred, _, _ = model.evaluate(v, q)
        batch_score = compute_score_with_logits(ans_pred, a.data).sum()
        score += batch_score

        upper_bound += (a.max(1)[0]).sum()
        num_data += ans_pred.size(0)

        if i % 100 == 0:
            print('Batch: [%d/%d]\nQuestion: %s\t Answer Prediction: %s\n'
                 'Explain GT:          %s\nExplain Prediction:  %s\n' %
                 (i, len(dataloader),
                  ' '.join([dataloader.dataset.question_dictionary.idx2word[w] for w in
                            q[0].data.tolist() if w > 0]),
                  dataloader.dataset.label2ans[ans_pred[0].max(0)[1].data[0]],
                  ' '.join([dataloader.dataset.explanation_dictionary.idx2word[w] for w in
                            c[0].tolist() if w > 2]),
                  ' '.join([dataloader.dataset.explanation_dictionary.idx2word[w] for w in
                            exp_pred[0].data.tolist() if w > 2])))

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    logger.write('G epoch %d, time: %.2f' % (epoch, time.time()-t))
    logger.write('\ttrain_score: %.2f' % (train_score[0]))
    logger.write('\teval score: %.2f (%.2f)' % (100 * score, 100 * upper_bound))
    return score
