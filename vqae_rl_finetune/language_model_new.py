import torch
import torch.nn as nn
import torch.nn.functional as F
import gumbel_trick as gumbel
from torch.autograd import Variable
import numpy as np
import pdb

class WordEmbedding(nn.Module):
    """Word Embedding

    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """
    def __init__(self, ntoken, emb_dim, dropout):
        super(WordEmbedding, self).__init__()
        self.emb = nn.Embedding(ntoken, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.ntoken = ntoken
        self.emb_dim = emb_dim

    def init_embedding(self, np_file):
        weight_init = torch.from_numpy(np.load(np_file))
        assert weight_init.shape == (self.ntoken, self.emb_dim)
        self.emb.weight.data[:self.ntoken] = weight_init

    def forward(self, x):
        emb = self.emb(x)
        emb = self.dropout(emb)
        return emb


class QuestionEmbedding(nn.Module):
    def __init__(self, in_dim, num_hid, nlayers, bidirect, dropout, rnn_type='GRU'):
        """Module for question embedding
        """
        super(QuestionEmbedding, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU

        self.rnn = rnn_cls(
            in_dim, num_hid, nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True)

        self.in_dim = in_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirect)

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid)
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(*hid_shape).zero_()),
                    Variable(weight.new(*hid_shape).zero_()))
        else:
            return Variable(weight.new(*hid_shape).zero_())

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)

        if self.ndirections == 1:
            return output[:, -1]

        forward_ = output[:, -1, :self.num_hid]
        backward = output[:, 0, self.num_hid:]
        return torch.cat((forward_, backward), dim=1)

    def forward_all(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)
        return output


class ExplainEmbedding(nn.Module):
    def __init__(self, w_emb, in_dim, num_hid, nlayers, bidirect, dropout, rnn_type='GRU'):
        super(ExplainEmbedding, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU

        self.rnn = rnn_cls(
            in_dim, num_hid, nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True)

        self.emb = w_emb
        self.dropout = nn.Dropout(0.0)
        self.in_dim = in_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirect)

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid)
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(*hid_shape).zero_()),
                    Variable(weight.new(*hid_shape).zero_()))
        else:
            return Variable(weight.new(*hid_shape).zero_())

    def sample_gumbel(self, logits, temperature=1.0):
        x = gumbel.gumbel_softmax(logits, temperature)
        x = torch.matmul(x, self.emb.weight)
        x = self.dropout(x)
        return x

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)

        if self.ndirections == 1:
            return output[:, -1]

        forward_ = output[:, -1, :self.num_hid]
        backward = output[:, 0, self.num_hid:]
        return torch.cat((forward_, backward), dim=1)


class SATDecoder(nn.Module):

    def __init__(self, w_emb, vis_dim, mm_dim, embed_dim, att_dim, decode_dim, vocab_size, num_layers=1, dropout_ratio=0.5):
        super(SATDecoder, self).__init__()

        self.embed_dim = embed_dim
        self.vis_dim = vis_dim
        self.att_dim = att_dim
        self.decode_dim = decode_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio

        #self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed = w_emb
        self.dropout = nn.Dropout(dropout_ratio) if dropout_ratio < 1 else None
        self.init_h = nn.Linear(mm_dim, decode_dim)
        self.init_c = nn.Linear(mm_dim, decode_dim)
        self.lstm_cell = nn.LSTMCell(embed_dim + vis_dim, decode_dim, num_layers)
        self.fc_dropout = nn.Dropout(dropout_ratio) if dropout_ratio < 1 else None
        self.fc_out = nn.Linear(decode_dim, vocab_size)

        # attention
        self.att_vw = nn.Linear(self.vis_dim, self.att_dim, bias=False)
        self.att_hw = nn.Linear(self.decode_dim, self.att_dim, bias=False)
        self.att_bias = nn.Parameter(torch.zeros(36))
        self.att_w = nn.Linear(self.att_dim, 1, bias=False)

    def init_embedding(self, np_file):
        weight_init = torch.from_numpy(np.load(np_file))
        assert weight_init.shape == (self.vocab_size, self.embed_dim)
        self.embed.weight.data[:self.vocab_size] = weight_init

    def _attention_layer(self, features, hiddens):
        """
        :param features:  batch_size  * 196 * 512
        :param hiddens:  batch_size * decode_dim
        :return:
        """
        att_fea = self.att_vw(features)
        # N-L-D
        att_h = self.att_hw(hiddens).unsqueeze(1)
        # N-1-D
        att_full = nn.ReLU()(att_fea + att_h + self.att_bias.view(1, -1, 1))
        att_out = self.att_w(att_full).squeeze(2)
        alpha = nn.Softmax()(att_out)
        # N-L
        context = torch.sum(features * alpha.unsqueeze(2), 1)
        return context, alpha

    def forward(self, visual, joint, captions, lengths, max_len):
        """
        :param features: batch_size * 196 * 512
        :param captions: batch_size * time_steps
        :param lengths:
        :return:
        """
        time_step = max_len-1

        h0, c0 = self.get_start_states(joint)
        outputs = []
        for i in range(time_step):
            it = captions[:, i].clone()
            if i >= 1 and captions[:, i].data.sum() == 0:
                break
            xt = self.embed(it)
            feas, alpha = self._attention_layer(visual, h0)
            input = torch.cat([feas, xt], 1)
            h0, c0 = self.lstm_cell(input, (h0, c0))
            output = self.fc_out(self.fc_dropout(h0))
            outputs.append(output)

        return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

    def sample(self, visual, joint, max_len=18):
        # greedy sample
        batch_size = visual.size(0)

        sampled_ids = []
        alphas = [0]

        start_var = Variable(torch.ones(batch_size, 1).cuda().long())
        words = self.embed(start_var).squeeze(1)
        h0, c0 = self.get_start_states(joint)
        word_flag = Variable(torch.ones(batch_size).cuda())

        for step in range(max_len):
            if step != 0:
                word_flag = (predicted!=0)*(predicted!=2)
            feas, alpha = self._attention_layer(visual, h0)
            alphas.append(alpha)
            inputs = torch.cat([feas, words], 1)
            h0, c0 = self.lstm_cell(inputs, (h0, c0))
            output = self.fc_out(h0)
            predicted = output.max(1)[1] * word_flag.long()
            sampled_ids.append(predicted.unsqueeze(1))
            words = self.embed(predicted)

        sampled_ids = torch.cat(sampled_ids, 1)
        return sampled_ids.squeeze()

    def get_start_states(self, joint_repr):
        h0 = self.init_h(joint_repr)
        c0 = self.init_c(joint_repr)
        return h0, c0


class STDecoder(nn.Module):

    def __init__(self, vis_dim, embed_dim, decode_dim, vocab_size, num_layers=1, dropout_ratio=0.5):
        super(STDecoder, self).__init__()

        self.embed_dim = embed_dim
        self.vis_dim = vis_dim
        self.decode_dim = decode_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        #self.embed = w_emb
        self.dropout = nn.Dropout(dropout_ratio) if dropout_ratio < 1 else None
        self.image_proj = nn.Linear(decode_dim, embed_dim)
        self.lstm_cell = nn.LSTMCell(embed_dim, decode_dim, num_layers)
        self.fc_dropout = nn.Dropout(dropout_ratio) if dropout_ratio < 1 else None
        self.fc_out = nn.Linear(decode_dim, vocab_size)

    def init_embedding(self, np_file):
        weight_init = torch.from_numpy(np.load(np_file))
        assert weight_init.shape == (self.vocab_size, self.embed_dim)
        self.embed.weight.data[:self.vocab_size] = weight_init

    def forward(self, joint, captions, lengths, max_len):
        """
        :param features: batch_size * 196 * 512
        :param captions: batch_size * time_steps
        :param lengths:
        :return:
        """
        time_step = max_len-1
        batch_size = joint.size(0)
        h0 = Variable(torch.zeros(batch_size, self.decode_dim).cuda())
        c0 = Variable(torch.zeros(batch_size, self.decode_dim).cuda())
        input = self.image_proj(joint)

        h0, c0 = self.lstm_cell(input, (h0, c0))
        #h0, c0 = self.get_start_states(joint)
        outputs = []
        for i in range(time_step):
            it = captions[:, i].clone()
            if i >= 1 and captions[:, i].data.sum() == 0:
                break
            xt = self.embed(it)
            h0, c0 = self.lstm_cell(xt, (h0, c0))
            output = F.log_softmax(self.fc_out(self.fc_dropout(h0)), dim=1)
            outputs.append(output)

        return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

    def sample(self, joint, max_len=18, opt={}):
        # greedy sample
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1:
            return self.sample_beam(joint, max_len, opt)

        batch_size = joint.size(0)
        sampled_ids = []

        start_var = Variable(torch.ones(batch_size).cuda().long())
        words = self.embed(start_var)

        h0 = Variable(torch.zeros(batch_size, self.decode_dim).cuda())
        c0 = Variable(torch.zeros(batch_size, self.decode_dim).cuda())
        input = self.image_proj(joint)

        h0, c0 = self.lstm_cell(input, (h0, c0))
        #h0, c0 = self.get_start_states(joint)
        word_flag = Variable(torch.ones(batch_size).cuda())
        outputs = []

        for step in range(max_len):
            if step != 0:
                word_flag = (predicted!=0)*(predicted!=2)
                if word_flag.data.sum() == 0:
                    break
            h0, c0 = self.lstm_cell(words, (h0, c0))
            output = F.log_softmax(self.fc_out(h0), dim=1)
            if sample_max:
                seqLogProbs, predicted = torch.max(output, 1)
                predicted = predicted.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(output).cpu()
                else:
                    prob_prev = torch.exp(torch.div(output, temperature)).cpu()
                predicted = torch.multinomial(prob_prev, 1).cuda()
                seqLogProbs = output.gather(1, predicted)
                predicted = predicted.view(-1).long()

            outputs.append(seqLogProbs)
            predicted = predicted * word_flag.long()
            sampled_ids.append(predicted.unsqueeze(1))
            words = self.embed(predicted)

        sampled_ids = torch.cat(sampled_ids, 1)
        return torch.cat([_.unsqueeze(1) for _ in outputs], 1), sampled_ids.squeeze()

    def _sample_beam(self, joint, max_len=18, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = joint.size(0)

        assert beam_size <= self.vocab_size, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(max_len, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(max_len, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            for t in range(2):
                if t == 0:
                    input = self.image_proj(joint[k:k+1]).expand(beam_size, self.input_encoding_size)
                elif t == 1: # input <bos>
                    it = joint.data.new(beam_size).long().zero_()
                    input = self.embed(it)

                h0, c0 = self.lstm_cell(input, (h0, c0))
                logprobs = F.log_softmax(self.fc_out(h0), dim=1)

            self.done_beams[k] = self.beam_search(state, logprobs, opt=opt)    #TODO: complete self.beam_search
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def sample_gumbel(self, joint, max_len=18, temperature=1.0):
        # greedy sample
        batch_size = joint.size(0)

        sampled_ids = []

        start_var = Variable(torch.ones(batch_size, 1).cuda().long())
        words = self.embed(start_var).squeeze(1)

        h0 = Variable(torch.zeros(batch_size, self.decode_dim).cuda())
        c0 = Variable(torch.zeros(batch_size, self.decode_dim).cuda())
        input = self.image_proj(joint)

        h0, c0 = self.lstm_cell(input, (h0, c0))
        #h0, c0 = self.get_start_states(joint)
        word_flag = Variable(torch.ones(batch_size).cuda())
        outputs = []

        for step in range(max_len):
            if step != 0:
                word_flag = (predicted!=0)*(predicted!=2)
            inputs = words
            h0, c0 = self.lstm_cell(inputs, (h0, c0))
            output = self.fc_out(h0)
            outputs.append(output.max(1)[0])
            predicted = gumbel.gumbel_softmax_ind(output, temperature) * word_flag.long()
            sampled_ids.append(predicted.unsqueeze(1))
            words = self.embed(predicted)

        sampled_ids = torch.cat(sampled_ids, 1)
        return torch.cat([_.unsqueeze(1) for _ in outputs], 1), sampled_ids.squeeze()
