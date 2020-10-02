import torch
import torch.nn as nn
from torch.autograd import Variable
from attention import Attention, NewAttention
from language_model_new import WordEmbedding, QuestionEmbedding, ExplainEmbedding, SATDecoder, STDecoder
from classifier import SimpleClassifier
from fc import FCNet
from train import l2_normalize
import time
import pdb


class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

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
        logits = self.classifier(joint_repr)
        return logits


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
        #vqe_pred = self.generator.sample_gumbel(joint_repr)
        _, vqe_pred = self.generator.sample(joint_repr)
        return vqa_logits, vqe_pred


class VQAE2(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, generator, e_emb, T_vq, T_e):
        super(VQAE2, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.generator = generator
        self.e_emb = e_emb
        self.T_vq = T_vq
        self.T_e = T_e
        self.dropout = nn.Dropout(0.0)
        #self.concat_proj = nn.Linear(2048, 1024)

    def forward(self, v, q, gt, lengths, max_len):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]
        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        vq_repr = q_repr * v_repr

        #vqe_logits = self.generator(joint_vq, gt, lengths, max_len)
        vqe_logits = None

        sentence = self.e_emb.emb(gt)
        e_emb = self.e_emb(sentence)
        e_proj = self.T_e(e_emb)

        vq_proj = None
        #vq_proj = self.T_vq(vq_repr)
        #joint_vqe = vq_repr * vq_proj
        joint_vqe = vq_repr * e_proj

        vqa_logits = self.classifier(joint_vqe)

        return vqa_logits, vq_proj, e_proj

    def vq_projection(self, v, q, gt):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]
        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        vq_repr = q_repr * v_repr

        sentence = self.e_emb.emb(gt)
        e_emb = self.e_emb(sentence)
        e_proj = self.T_e.main[0](e_emb)
        # e feature normalize #
        # e_proj = l2_normalize(e_proj)

        vq_proj = nn.Sequential(*list(self.T_vq.main.children())[:-1])(vq_repr) # Without ReLU
        # vq projection normalize #
        # vq_proj = l2_normalize(vq_proj)

        return vq_proj, e_proj

    def evaluate(self, v, q, gt):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]
        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        vq_repr = q_repr * v_repr

        sentence = self.e_emb.emb(gt)
        e_emb = self.e_emb(sentence)
        e_proj = self.T_e(e_emb)

        joint_vqe = vq_repr * e_proj
        vqa_logits = self.classifier(joint_vqe)

        return vqa_logits

    def evaluate_projection(self, v, q, gt):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]
        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        vq_repr = q_repr * v_repr

        sentence = self.e_emb.emb(gt)
        e_emb = self.e_emb(sentence)
        e_proj = self.T_e(e_emb)
        # e feature normalize #
        #e_proj = l2_normalize(e_proj)

        vq_proj = self.T_vq(vq_repr)
        # vq projection normalize #
        # vq_proj = l2_normalize(vq_proj)
        joint_vqe = vq_repr * vq_proj

        vqa_logits = self.classifier(joint_vqe)

        return vqa_logits


class Split_VQAE(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, v_att_2, q_net_2, v_net_2, classifier, generator, e_emb, T_vq, T_e):
        super(Split_VQAE, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.v_att_2 = v_att_2
        self.q_net_2 = q_net_2
        self.v_net_2 = v_net_2
        self.classifier = classifier
        self.generator = generator
        self.e_emb = e_emb
        self.T_vq = T_vq
        self.T_e = T_e
        self.dropout = nn.Dropout(0.0)

    def forward(self, v, q, gt, lengths, max_len):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        att_2 = self.v_att_2(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]
        v_emb_2 = (att_2 * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        q_repr_2 = self.q_net(q_emb)
        v_repr_2 = self.v_net_2(v_emb_2)
        vq = q_repr * v_repr
        vq_2 = q_repr_2 * v_repr_2

        #vqe_logits = self.generator(joint_vq, gt, lengths, max_len)
        vqe_logits = None

        sentence = self.e_emb.emb(gt)
        e_emb = self.e_emb(sentence)
        e_proj = self.T_e(e_emb)
        vq_2 = self.T_vq(vq_2)
        joint_vqe = vq * vq_2

        vqa_logits = self.classifier(joint_vqe)

        return vqa_logits, vqe_logits, vq_2, e_proj

    def evaluate(self, v, q, gt):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        vq = q_repr * v_repr
        q_repr_2 = self.q_net_2(q_emb)
        v_repr_2 = self.v_net_2(v_emb)
        vq_2 = q_repr_2 * v_repr_2
        vq_2 = self.T_vq(vq_2)
        joint_vqe = vq * vq_2

        vqe_logits = None
        vqa_logits = self.classifier(joint_vqe)

        return vqa_logits, vqe_logits

def build_baseline(dataset, num_hid, att_dim, dec_dim):
    w_emb = WordEmbedding(dataset.question_dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)


def build_vqae_newatt(dataset, num_hid, att_dim, dec_dim):
    w_emb = WordEmbedding(dataset.question_dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    generator = STDecoder(
        dataset.v_dim, num_hid, 300, dec_dim,\
        dataset.explanation_dictionary.ntoken, 1, 0.5)
    return VQAE(w_emb, q_emb, v_att, q_net, v_net, classifier, generator)


def build_vqae2_newatt(dataset, num_hid, att_dim, dec_dim):
    w_emb = WordEmbedding(dataset.question_dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    generator = STDecoder(
        dataset.v_dim, num_hid, 300, dec_dim,\
        dataset.explanation_dictionary.ntoken, 1, 0.5)
    e_emb = ExplainEmbedding(generator.embed, 300, num_hid, 1, False, 0.0, 'GRU')
    T_vq = FCNet([num_hid, num_hid, num_hid])
    T_e = FCNet([e_emb.num_hid, num_hid])
    return VQAE2(w_emb, q_emb, v_att, q_net, v_net, classifier, generator, e_emb, T_vq, T_e)


def build_vqae3_split(dataset, num_hid, att_dim, dec_dim):
    w_emb = WordEmbedding(dataset.question_dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att_1 = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net_1 = FCNet([q_emb.num_hid, num_hid])
    v_net_1 = FCNet([dataset.v_dim, num_hid])
    v_att_2 = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net_2 = FCNet([q_emb.num_hid, num_hid])
    v_net_2 = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    generator = STDecoder(
        dataset.v_dim, num_hid, 300, dec_dim,\
        dataset.explanation_dictionary.ntoken, 1, 0.5)
    e_emb = ExplainEmbedding(generator.embed, 300, num_hid, 1, False, 0.0, 'GRU')
    T_vq = FCNet([num_hid, num_hid])
    T_e = FCNet([e_emb.num_hid, num_hid])
    return Split_VQAE(w_emb, q_emb, v_att_1, q_net_1, v_net_1, v_att_2, q_net_2, v_net_2, classifier, generator, e_emb, T_vq, T_e)
