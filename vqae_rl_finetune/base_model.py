import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model_new import WordEmbedding, QuestionEmbedding, ExplainEmbedding, SATDecoder, STDecoder
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
        #vqe_pred = self.generator.sample_gumbel(joint_repr)
        _, vqe_pred = self.generator.sample(joint_repr)
        return vqa_logits, vqe_pred


class VQAE2(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, generator, e_emb, e_net):
        super(VQAE2, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.generator = generator
        self.e_emb = e_emb
        self.e_net = e_net
        self.dropout = nn.Dropout(0.0)

    def forward(self, v, q, gt, lengths, max_len, t):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]
        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_vq = q_repr * v_repr

        vqe_logits = self.generator(joint_vq, gt, lengths, max_len)
        sentence = self.e_emb.sample_gumbel(vqe_logits, temperature=t)
        e_emb = self.e_emb(sentence)
        #sentence = self.e_emb.emb(gt)
        #e_emb = self.e_emb(sentence)
        e_repr = self.e_net(e_emb)
        joint_eq = q_repr * e_repr
        vqa_logits_vq = self.classifier(joint_vq)
        vqa_logits_eq = self.classifier(joint_eq)
        vqa_logits = [vqa_logits_vq, vqa_logits_eq]

        return vqa_logits, vqe_logits

    def evaluate(self, v, q, gt=None, t=1.0):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_vq = q_repr * v_repr

        _, vqe_pred = self.generator.sample(joint_vq)
        if gt is None:
            sentence = self.e_emb.emb(vqe_pred)
            e_emb = self.e_emb(sentence)
        else:
            sentence = self.e_emb.emb(gt)
            e_emb = self.e_emb(sentence)
        e_repr = self.e_net(e_emb)
        joint_eq = q_repr * e_repr
        vqa_logits_vq = self.classifier(joint_vq)
        vqa_logits_eq = self.classifier(joint_eq)
        vqa_logits = [vqa_logits_vq, vqa_logits_eq]
        return vqa_logits, vqe_pred

    def explain_branch(self, v, q, gt, lengths, max_len):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # [batch, q_dim]
        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1)  # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_vq = q_repr * v_repr
        #joint_vq = joint_vq.detach()

        vqe_logits = self.generator(joint_vq, gt, lengths, max_len)
        return vqe_logits

    def explain_sample(self, v, q, opt={}):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1)  # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        joint_repr = joint_repr.detach()
        vqe_logits, vqe_pred = self.generator.sample(joint_repr, opt=opt)
        return vqe_pred, vqe_logits


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


def build_vqae_newatt(dataset, num_hid, att_dim, dec_dim):
    w_emb = WordEmbedding(dataset.question_dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    generator = STDecoder(
        dataset.v_dim, 300, dec_dim,\
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
        dataset.v_dim, 300, dec_dim,\
        dataset.explanation_dictionary.ntoken, 1, 0.5)
    e_emb = ExplainEmbedding(generator.embed, 300, num_hid, 1, False, 0.0)
    e_net = FCNet([e_emb.num_hid, num_hid])
    return VQAE2(w_emb, q_emb, v_att, q_net, v_net, classifier, generator, e_emb, e_net)
