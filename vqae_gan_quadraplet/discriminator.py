from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn.utils.weight_norm import weight_norm
from torch.autograd import Variable
import os
import pdb

def compute_bc_accuracy(logits, labels):
    pred = logits >= 0.5
    num_correct = (pred == labels.byte().cuda()).sum()
    acc = (float(num_correct) * 100 / len(labels))
    return acc


def dTrain(model, real, fake, e=None):
    if e is None:
        r_pred = model(real)
        f_pred = model(fake)
    else:
        r_pred = model(real, e)
        f_pred = model(fake, e)
    if not isinstance(real, list):
        real = [real]; fake = [fake]
    r_label = Variable(torch.ones(real[0].size(0), 1)).cuda()
    f_label = Variable(torch.zeros(fake[0].size(0), 1)).cuda()
    r_loss = model.criterion(r_pred, r_label)
    f_loss = model.criterion(f_pred, f_label)
    loss = r_loss + f_loss
    return r_pred, f_pred, r_label, f_label, loss

def dAdvTrain(model, fake, e=None):
    if e is None:
        f_pred = model(fake)
    else:
        f_pred = model(fake, e)
    if not isinstance(fake, list):
        fake = [fake]
    r_label = Variable(torch.ones(fake[0].size(0), 1)).cuda()
    loss = model.criterion(f_pred, r_label)
    return loss

def dEval(model, real, fake, e=None):
    if e is None:
        r_pred = model(real)
        f_pred = model(fake)
    else:
        r_pred = model(real, e)
        f_pred = model(fake, e)
    if not isinstance(real, list):
        real = [real]; fake = [fake]
    r_label = Variable(torch.ones(real[0].size(0), 1)).cuda()
    f_label = Variable(torch.zeros(fake[0].size(0), 1)).cuda()
    return r_pred, f_pred, r_label, f_label


class RFdiscriminator(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, in_dim):
        super(RFdiscriminator, self).__init__()

        layers = []
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())

        self.main = nn.Sequential(*layers)
        self.criterion = nn.BCELoss()

    def forward(self, x):
        return self.main(x)


class VQAEdiscriminator(nn.Module):
    def __init__(self, v_dim, q_dim, embed_dim, e_dim, mm_dim):
        super(VQAEdiscriminator, self).__init__()

        a_emb_file = os.path.join('data', 'VQA2', 'glove6b_init_300d_answer.npy')
        weight_init = torch.from_numpy(np.load(a_emb_file))
        ntoken = weight_init.size(0)
        self.v_proj = nn.Linear(v_dim, mm_dim)
        self.q_proj = nn.Linear(q_dim, mm_dim)
        self.emb = nn.Embedding(ntoken, embed_dim)
        self.emb.weight.data[:ntoken] = weight_init
        self.a_proj = nn.Linear(embed_dim, mm_dim)
        self.e_proj = nn.Linear(e_dim, mm_dim)

        self.classifier = nn.Linear(mm_dim, 1)
        self.tanh   = nn.Tanh()
        self.sig    = nn.Sigmoid()
        self.criterion = nn.BCELoss()

    def forward(self, inputs):
        v = inputs[0]
        q = inputs[1]
        a = inputs[2]
        e = inputs[3]

        v = v.mean(1)
        v = self.v_proj(v)
        v = self.tanh(v)
        q = self.q_proj(q)
        q = self.tanh(q)
        pa = a.clone()
        pa[:,0] = pa[:,0] + (pa.sum(1)==0).float()
        sa = torch.multinomial(pa, 1)
        sa = sa.squeeze()
        ea = self.emb(sa)
        #ea = self.dropout(ea)
        ea = self.a_proj(ea.detach())
        ea = self.tanh(ea)
        e = self.e_proj(e)
        e = self.tanh(e)
        vqae = v * q * ea * e
        logit = self.classifier(vqae)
        logit = self.sig(logit)
        return logit


class VEdiscriminator(nn.Module):
    def __init__(self, v_dim, e_dim, mm_dim):
        super(VEdiscriminator, self).__init__()

        self.v_proj = nn.Linear(v_dim, mm_dim)
        self.e_proj = nn.Linear(e_dim, mm_dim)
        self.classifier = nn.Linear(mm_dim, 1)
        self.tanh   = nn.Tanh()
        self.sig    = nn.Sigmoid()
        self.criterion = nn.BCELoss()

    def forward(self, v, e):
        v  = v.mean(1)
        v  = self.v_proj(v)
        v  = self.tanh(v)
        e  = self.e_proj(e)
        e  = self.tanh(e)
        #e  = e.unsqueeze(1).expand_as(v)
        ve = v * e
        #ve = ve.sum(1)
        logit = self.classifier(ve)
        logit = self.sig(logit)
        return logit


class QEdiscriminator(nn.Module):
    def __init__(self, q_dim, e_dim, mm_dim):
        super(QEdiscriminator, self).__init__()

        layers = []
        layers.append(nn.Linear(q_dim + e_dim, mm_dim))
        layers.append(nn.Tanh())
        layers.append(nn.Linear(mm_dim, 1))
        layers.append(nn.Sigmoid())

        self.main = nn.Sequential(*layers)
        self.criterion = nn.BCELoss()

    def forward(self, q, e):
        x = torch.cat([q, e], 1)
        return self.main(x)


class AEdiscriminator(nn.Module):
    def __init__(self, embed_dim, e_dim, mm_dim, dropout):
        super(AEdiscriminator, self).__init__()

        a_emb_file = os.path.join('data', 'VQA2', 'glove6b_init_300d_answer.npy')
        weight_init = torch.from_numpy(np.load(a_emb_file))
        ntoken = weight_init.size(0)
        self.emb = nn.Embedding(ntoken, embed_dim)
        self.emb.weight.data[:ntoken] = weight_init
        self.dropout = nn.Dropout(dropout)

        self.a_proj = nn.Linear(embed_dim, mm_dim)
        self.e_proj = nn.Linear(e_dim, mm_dim)
        self.classifier = nn.Linear(mm_dim, 1)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.criterion = nn.BCELoss()

    def forward(self, a, e):
        pa = a.clone()
        pa[:,0] = pa[:,0] + (pa.sum(1)==0).float()
        sa = torch.multinomial(pa, 1)
        sa = sa.squeeze()
        ea = self.emb(sa)
        ea = self.dropout(ea)
        ea = self.a_proj(ea.detach())
        ea = self.tanh(ea)
        e = self.e_proj(e)
        e = self.tanh(e)
        ae = ea * e
        logit = self.classifier(ae)
        logit = self.sig(logit)
        return logit


class discCrit(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self):
        super(discCrit, self).__init__()

        self.main = nn.BCELoss()

    def forward(self, x):
        return self.main(x)

