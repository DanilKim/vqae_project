import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model_new import WordEmbedding, QuestionEmbedding, ExplainEmbedding, SATDecoder, STDecoder
from classifier import SimpleClassifier
from fc import FCNet
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
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, generator, e_net):
        super(VQAE, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.generator = generator
        self.e_net = e_net
        self.dropout = nn.Dropout(0.0)

    def generate(self, v, q):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        _, vqe_pred, last_hidden = self.generator.sample(joint_repr)

        return vqe_pred, last_hidden

    def classify(self, v, q):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_vq = q_repr * v_repr

        vqa_logits = self.classifier(joint_vq)
        return vqa_logits

    def forward(self, v, q, gt, lengths, max_len):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_vq = q_repr * v_repr

        vqe_logits, last_hidden = self.generator(joint_vq, gt, lengths, max_len)
        e_repr = self.e_net(last_hidden)
        joint_vqe = joint_vq * e_repr
        vqa_logits = self.classifier(joint_vqe)

        return vqa_logits, vqe_logits, q_emb, last_hidden

    def evaluate(self, v, q):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_vq = q_repr * v_repr

        vqe_logits, vqe_pred, last_hidden = self.generator.sample(joint_vq)
        e_repr = self.e_net(last_hidden)
        joint_vqe = joint_vq * e_repr
        vqa_logits = self.classifier(joint_vqe)

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

    def generate(self, v, q, gt=None, lengths=None, max_len=None, t=1.0, tf=False):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_vq = q_repr * v_repr

        if tf:
            #assert (gt!=None and lengths!=None and max_len!=None)
            vqe_logits, vqe_pred = self.generator(joint_vq, gt, lengths, max_len)
        else:
            vqe_logits, vqe_pred, _ = self.generator.sample(joint_vq)

        return vqe_logits, vqe_pred, joint_vq

    def dec_enc(self, vqe_logits, vqe_pred=None, t=1.0):
        if vqe_pred is not None:
            sentence = self.e_emb.emb(vqe_pred)
            return self.e_emb(sentence)
        sentence = self.e_emb.sample_gumbel(vqe_logits, temperature=t)
        e_emb = self.e_emb(sentence)
        return e_emb

    def classify(self, v, q):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_vq = q_repr * v_repr

        vqa_logits = self.classifier(joint_vq)
        return vqa_logits

    def forward(self, v, q, gt, lengths, max_len, t=1.0):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]
        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_vq = q_repr * v_repr

        real_vqe_logits, _ = self.generator(joint_vq, gt, lengths, max_len)
        fake_vqe_logits, _, _ = self.generator.sample(joint_vq)

        real_sentence = self.e_emb.sample_gumbel(real_vqe_logits, temperature=t)
        fake_sentence = self.e_emb.sample_gumbel(fake_vqe_logits, temperature=t)
        real_hidden = self.e_emb(real_sentence)
        fake_hidden = self.e_emb(fake_sentence)
        e_repr = self.e_net(fake_hidden)
        joint_vqe = joint_vq * e_repr
        vqa_logits = self.classifier(joint_vqe)

        return vqa_logits, real_vqe_logits, joint_vq, real_hidden, fake_hidden

    def evaluate(self, v, q):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_vq = q_repr * v_repr

        vqe_logits, vqe_pred, _ = self.generator.sample(joint_vq)
        #sentence = self.e_emb.emb(vqe_pred)
        sentence = self.e_emb.sample_gumbel(vqe_logits)
        e_emb = self.e_emb(sentence)
        e_repr = self.e_net(e_emb)
        joint_vqe = joint_vq * e_repr
        vqa_logits = self.classifier(joint_vqe)
        return vqa_logits, vqe_pred, joint_vq, e_emb


def build_baseline0(dataset, num_hid):
    w_emb = WordEmbedding(dataset.question_dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
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
    e_net = FCNet([dec_dim, num_hid])
    return VQAE(w_emb, q_emb, v_att, q_net, v_net, classifier, generator, e_net)


def build_vqae_newatt_2(dataset, num_hid, att_dim, dec_dim):
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
    e_emb = ExplainEmbedding(generator.embed, 300, num_hid, 1, False, 0.0, 'LSTM')
    e_net = FCNet([e_emb.num_hid, num_hid])
    return VQAE2(w_emb, q_emb, v_att, q_net, v_net, classifier, generator, e_emb, e_net)


def build_VQA_newatt_2(dataset, num_hid, att_dim, dec_dim):
    w_emb = WordEmbedding(dataset.question_dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)


#def build_VQA_newatt_2(dataset, num_hid, att_dim, dec_dim):
#    w_emb = WordEmbedding(dataset.question_dictionary.ntoken, 300, 0.0)
#    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
#    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
#    q_net = FCNet([q_emb.num_hid, num_hid])
#    v_net = FCNet([dataset.v_dim, num_hid])
#    classifier = SimpleClassifier(
#        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
#    return VQAE2(w_emb, q_emb, v_att, q_net, v_net, classifier, None, None, None)


def build_VQE_newatt_2(dataset, num_hid, att_dim, dec_dim):
    w_emb = WordEmbedding(dataset.question_dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    generator = STDecoder(
        dataset.v_dim, num_hid, 300, dec_dim,\
        dataset.explanation_dictionary.ntoken, 1, 0.5)
    return VQAE2(w_emb, q_emb, v_att, q_net, v_net, None, generator, None, None)
