import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class WordEmbedding(nn.Module):
    """Word Embedding

    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """
    def __init__(self, ntoken, emb_dim, dropout):
        super(WordEmbedding, self).__init__()
        self.emb = nn.Embedding(ntoken+1, emb_dim, padding_idx=ntoken)
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


class Decoder(nn.Module):

    def __init__(self, vis_dim, mm_dim, embed_dim, att_dim, decode_dim, vocab_size, num_layers=1, dropout_ratio=0.5):
        super(Decoder, self).__init__()

        self.embed_dim = embed_dim
        self.vis_dim = vis_dim
        self.att_dim = att_dim
        self.decode_dim = decode_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
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
        batch_size = captions.size(0)
        time_step = max_len-1
        vocab_size = self.vocab_size
        embed = self.embed
        dropout = self.dropout
        attention_layer = self._attention_layer
        lstm_cell = self.lstm_cell
        fc_dropout = self.fc_dropout
        fc_out = self.fc_out

        word_embeddings = embed(captions)
        word_embeddings = dropout(word_embeddings) if dropout is not None else word_embeddings
        feas = torch.mean(visual, 1)  # batch_size * 512
        h0, c0 = self.get_start_states(joint)

        predicts = Variable(torch.zeros(batch_size, time_step, vocab_size).cuda())

        for step in range(time_step):
            batch_size = sum(i > step for i in lengths)
            if step != 0:
                feas, alpha = attention_layer(visual[:batch_size, :], h0[:batch_size, :])
            words = (word_embeddings[:batch_size, step, :]).squeeze(1)
            inputs = torch.cat([feas, words], 1)
            h0, c0 = lstm_cell(inputs, (h0[:batch_size, :], c0[:batch_size, :]))
            outputs = fc_out(fc_dropout(h0)) if fc_dropout is not None else fc_out(h0)
            predicts[:batch_size, step, :] = outputs

        return predicts

    def sample(self, visual, joint, max_len=18):
        # greedy sample
        embed = self.embed
        lstm_cell = self.lstm_cell
        fc_out = self.fc_out
        attend = self._attention_layer
        batch_size = visual.size(0)

        sampled_ids = []
        alphas = [0]

        words = embed(Variable(torch.ones(batch_size, 1).cuda().long())).squeeze(1)
        h0, c0 = self.get_start_states(joint)
        feas = torch.mean(visual, 1) # convert to batch_size*512
        word_flag = Variable(torch.ones(batch_size).cuda())

        for step in range(max_len):
            if step != 0:
                feas, alpha = attend(visual, h0)
                alphas.append(alpha)
                word_flag = (predicted!=0)*(predicted!=2)
            inputs = torch.cat([feas, words], 1)
            h0, c0 = lstm_cell(inputs, (h0, c0))
            outputs = fc_out(h0)
            predicted = outputs.max(1)[1] * word_flag.long()
            sampled_ids.append(predicted.unsqueeze(1))
            words = embed(predicted)

        sampled_ids = torch.cat(sampled_ids, 1)
        return sampled_ids.squeeze()

    def get_start_states(self, joint_repr):
        h0 = self.init_h(joint_repr)
        c0 = self.init_c(joint_repr)
        return h0, c0