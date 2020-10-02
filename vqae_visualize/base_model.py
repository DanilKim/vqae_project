import torch
import torch.nn as nn
from torch.autograd import Variable
from attention import Attention, NewAttention
from language_model_new import WordEmbedding, QuestionEmbedding, SATDecoder, STDecoder
from classifier import SimpleClassifier
from fc import FCNet
import pdb


class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net):#, classifier):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        #self.classifier = classifier

    def forward(self, v, q):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        #logits = self.classifier(joint_repr)
        return joint_repr #logits, joint_repr


class VQAE(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, generator):
        super(VQAE, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.generator = generator

    def forward(self, v, q, gt, lengths, max_len):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        vqa_logits = self.classifier(joint_repr)
        vqe_logits = self.generator(joint_repr, gt, lengths, max_len)

        return vqa_logits, vqe_logits

    def evaluate(self, v, q):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        vqa_logits = self.classifier(joint_repr)
        vqe_pred = self.generator.sample(joint_repr)
        return vqa_logits, vqe_pred


class VQE(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, generator):
        super(VQE, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.generator = generator

    def forward(self, v, q, gt, lengths, max_len):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        vqe_logits, alphas = self.generator(v, joint_repr, gt, lengths, max_len)

        return vqe_logits, alphas

    def evaluate(self, v, q):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        vqe_pred, alphas = self.generator.sample(v, joint_repr)
        return vqe_pred, alphas, q_emb


class LSTM_VQA(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, generator, att_emb, classifier):
        super(LSTM_VQA, self).__init__()
        self.generator = VQE(w_emb, q_emb, v_att, q_net, v_net, generator)
        self.att_emb = att_emb
        self.att_net = FCNet([1024, 1024])
        self.q_net = FCNet([1024, 1024])
        self.classifier = classifier

    def forward(self, v, q):
        vqe_pred, alphas, q_emb = self.generator.evaluate(v, q)
        lengths = (vqe_pred > 0).sum(1)
        lengths, sort_ind = lengths.sort(dim=0, descending=True)
        time_step = int(lengths[0])
        batch_size = v.size(0)

        v = v[sort_ind]
        q = q[sort_ind]
        vqe_pred = vqe_pred[:, :time_step][sort_ind]
        alphas = alphas[:, :time_step][sort_ind]
        alphas = alphas * (vqe_pred.unsqueeze(1).expand_as(alphas) > 0).float()

        max_att, max_idx = alphas.max(1)

        #h0 = Variable(torch.zeros(1, batch_size, 1024).cuda())
        h0 = Variable(torch.zeros(batch_size, 1024).cuda())
        last_hidden = Variable(torch.zeros(batch_size, 512).cuda())
        #outputs = []

        #input = v[[range(batch_size)] * time_step, max_idx.transpose(0,1)].transpose(0,1)
        #_, last_hidden = self.att_emb(input, h0)

        for t in range(time_step):
            input = v[range(batch_size), max_idx[:,t]]
            input = input * max_att[:, t].unsqueeze(1).expand_as(input)
            pdb.set_trace()
            h0 = self.att_emb(input, h0)
            if (t == lengths.data-1).sum() > 0:
                last_idx = (t == lengths.data-1).nonzero().squeeze().cuda()
                last_hidden[last_idx] = h0[last_idx]

        q_repr = self.q_net(q_emb)
        v_repr = self.att_net(last_hidden)
        joint_repr = q_repr * v_repr.squeeze()

        vqa_logits = self.classifier(joint_repr)
        return vqa_logits, vqe_pred, alphas


def build_baseline0(dataset, num_hid):
    w_emb = WordEmbedding(dataset.question_dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)


def build_baseline0_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.question_dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)


def build_vqe_newatt(dataset, num_hid, att_dim, dec_dim):
    w_emb = WordEmbedding(dataset.question_dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    generator = SATDecoder(
        dataset.v_dim, num_hid, 300, att_dim, dec_dim,\
        dataset.explanation_dictionary.ntoken, 1, 0.5)
    return VQE(w_emb, q_emb, v_att, q_net, v_net, generator)


def build_vqae_newatt(dataset, num_hid, att_dim, dec_dim):
    w_emb = WordEmbedding(dataset.question_dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    generator = SATDecoder(
        dataset.v_dim, num_hid, 300, att_dim, dec_dim,\
        dataset.explanation_dictionary.ntoken, 1, 0.5)
    return VQAE(w_emb, q_emb, v_att, q_net, v_net, classifier, generator)


def build_lstm_vqa(dataset, num_hid, att_dim, dec_dim):
    w_emb = WordEmbedding(dataset.question_dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    generator = SATDecoder(
        dataset.v_dim, num_hid, 300, att_dim, dec_dim,\
        dataset.explanation_dictionary.ntoken, 1, 0.5)
    #att_emb = nn.GRU(dataset.v_dim, num_hid, 1, False, batch_first=True)
    att_emb = nn.GRUCell(dataset.v_dim, num_hid)
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return LSTM_VQA(w_emb, q_emb, v_att, q_net, v_net, generator, att_emb, classifier)
