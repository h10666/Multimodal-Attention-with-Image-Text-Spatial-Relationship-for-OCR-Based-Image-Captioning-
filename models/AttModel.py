# This file contains Att2in2, AdaAtt, AdaAttMO, TopDown model

# AdaAtt is from Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning
# https://arxiv.org/abs/1612.01887
# AdaAttMO is a modified version with maxout lstm

# Att2in is from Self-critical Sequence Training for Image Captioning
# https://arxiv.org/abs/1612.00563
# In this file we only have Att2in2, which is a slightly different version of att2in,
# in which the img feature embedding and word embedding is the same as what in adaatt.

# TopDown is from Bottom-Up and Top-Down Attention for Image Captioning and VQA
# https://arxiv.org/abs/1707.07998
# However, it may not be identical to the author's architecture.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import misc.utils as utils
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from .CaptionModel import CaptionModel

def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0,len(indices)).type_as(inv_ix)
    return tmp, inv_ix

def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp

def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)

def _batch_gather(x, inds):
    assert x.dim() == 3
    batch_size = x.size(0)
    length = x.size(1)
    dim = x.size(2)
    x_flat = x.view(batch_size*length, dim)

    batch_offsets = torch.arange(batch_size, device=inds.device) * length
    if inds.dim() == 2:
        batch_offsets = batch_offsets.unsqueeze(-1)
    assert batch_offsets.dim() == inds.dim()
    inds_flat = batch_offsets + inds
    results = F.embedding(inds_flat, x_flat)
    return results

class OcrPtrNet(nn.Module):
    def __init__(self, hidden_size, query_key_size=None):
        super(OcrPtrNet, self).__init__()

        if query_key_size is None:
            query_key_size = hidden_size
        self.hidden_size = hidden_size
        self.query_key_size = query_key_size

        self.query = nn.Linear(hidden_size, query_key_size)
        self.key = nn.Linear(hidden_size, query_key_size)

    # def forward(self, query_inputs, key_inputs, attention_mask):
    #     extended_attention_mask = (1.0 - attention_mask) * -10000.0
    #     assert extended_attention_mask.dim() == 2
    #     extended_attention_mask = extended_attention_mask.unsqueeze(1)

    #     query_layer = self.query(query_inputs)
    #     if query_layer.dim() == 2:
    #         query_layer = query_layer.unsqueeze(1)
    #         squeeze_result = True
    #     else:
    #         squeeze_result = False
    #     key_layer = self.key(key_inputs)

    #     scores = torch.matmul(
    #         query_layer,
    #         key_layer.transpose(-1, -2)
    #     )
    #     scores = scores / math.sqrt(self.query_key_size)
    #     scores = scores + extended_attention_mask
    #     if squeeze_result:
    #         scores = scores.squeeze(1)

    #     return scores

    def forward(self, query_inputs, key_inputs, attention_mask):
        query_layer = self.query(query_inputs).unsqueeze(1)
        key_layer = self.key(key_inputs)

        scores = torch.matmul(
            query_layer,
            key_layer.transpose(-1, -2)
        )
        scores = scores.squeeze(1) / math.sqrt(self.query_key_size)
        # scores = scores * attention_mask

        return scores      

class AttModel(CaptionModel):
    def __init__(self, opt):
        super(AttModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.ocr_size = opt.ocr_size

        self.use_bn = getattr(opt, 'use_bn', 0)

        self.ss_prob = 0.0 # Schedule sampling probability

        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                nn.ReLU(),
                                nn.Dropout(self.drop_prob_lm))
        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Linear(self.att_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn==2 else ())))
        self.ocr_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size+300+604),) if self.use_bn else ())+
                                    (nn.Linear(self.att_feat_size+300+604, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn==2 else ())))
        self.ocr_box_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Linear(5, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn==2 else ())))

        self.logit_layers = getattr(opt, 'logit_layers', 1)
        if self.logit_layers == 1:
            self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
            self.logit2 = nn.Linear(self.ocr_size, self.ocr_size)
        else:
            self.logit = [[nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(), nn.Dropout(0.5)] for _ in range(opt.logit_layers - 1)]
            self.logit = nn.Sequential(*(reduce(lambda x,y:x+y, self.logit) + [nn.Linear(self.rnn_size, self.vocab_size + 1)]))
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.ocr_ptr_net = OcrPtrNet(self.rnn_size)
        self.ctx2att_2 = nn.Linear(self.rnn_size, self.att_hid_size)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, att_masks, ocr_feats, ocr_masks, ocr_relations):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats)

        ocr_feats = self.ocr_embed(ocr_feats[:,:,:-5]) + \
                    self.ocr_box_embed(ocr_feats[:,:,-5:])

        rel_mask = (ocr_relations == 2) + (ocr_relations == 4)
        vocab_mask = torch.zeros((att_feats.shape[0], self.vocab_size+1, 1)).expand(-1,-1,self.ocr_size)
        vocab_rel_mask = torch.cat([vocab_mask.cuda(), rel_mask.float()], dim=1)
        p_ocr_feats = self.ctx2att_2(ocr_feats)

        return fc_feats, att_feats, p_att_feats, att_masks, ocr_feats, p_ocr_feats, ocr_masks, vocab_rel_mask

    def _forward(self, ocr_feats, ocr_masks, ocr_relations, fc_feats, att_feats, seq, att_masks=None):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        outputs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size+1+ocr_feats.size(1))

        # Prepare the features
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, p_ocr_feats, pp_ocr_feats, p_ocr_masks, vocab_rel_mask = self._prepare_feature(fc_feats, att_feats, att_masks, ocr_feats, ocr_masks, ocr_relations)
        # pp_att_feats is used for attention, we cache it in advance to reduce computation cost

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    #prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                    #it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                    # prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                    prob_prev = torch.exp(outputs[:, i-1].detach()) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()          
            # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            for bi in range(batch_size):
                if bi == 0:
                    vr_mask = vocab_rel_mask[0,it[0]].unsqueeze(0)
                else:
                    vr_mask = torch.cat([vr_mask, vocab_rel_mask[bi,it[bi]].unsqueeze(0)], dim=0)

            output, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, p_ocr_feats, pp_ocr_feats, p_ocr_masks, vr_mask, state)
            outputs[:, i] = output

        return outputs

    def word_embedding(self, vocab_emb, ocr_emb, it):
        batch_size = it.size(0)
        vocab_num = vocab_emb.size(0)

        assert vocab_emb.size(-1) == ocr_emb.size(-1), "Vocab embedding and ocr embedding do not match."
        vocab_emb = vocab_emb.unsqueeze(0).expand(batch_size, -1, -1)
        vocab_ocr_emb_cat = torch.cat([vocab_emb, ocr_emb], dim=1)
        word_emb = _batch_gather(vocab_ocr_emb_cat, it)

        return word_emb

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, ocr_feats, p_ocr_feats, ocr_masks, vr_mask, state):
        # 'it' contains a word index
        # xt = self.embed(it)
        xt = self.word_embedding(self.embed[0].weight, ocr_feats, it)

        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, ocr_feats, p_ocr_feats, ocr_masks, state, att_masks)
        fixed_scores = self.logit(output)  # (b, vocab_size)
        dynamic_ocr_scores = self.ocr_ptr_net(state[0][2], ocr_feats, ocr_masks) # (b, ocr_size)
        dynamic_ocr_scores = self.logit2(dynamic_ocr_scores)
        scores = torch.cat([fixed_scores, dynamic_ocr_scores], dim=-1)

        probs = F.softmax(scores, dim=1)
        
        # zeroed_masks = torch.zeros_like(fixed_scores)
        # entended_vr_mask = torch.cat((zeroed_masks, vr_mask), dim = -1)
        # entended_vr_mask = entended_vr_mask + torch.ones_like(entended_vr_mask)
        # probs = probs * entended_vr_mask
        # probs = probs/probs.sum(1,keepdim=True)

        logprobs = torch.log(probs)
        fixed_masks = torch.ones_like(fixed_scores)
        both_masks = torch.cat((fixed_masks, ocr_masks), dim = -1)
        both_masks = (1.0 - both_masks) * -10000.0
        logprobs = logprobs + both_masks

        return logprobs, state

    def get_logprobs_state_test(self, it, fc_feats, att_feats, p_att_feats, att_masks, ocr_feats, p_ocr_feats, ocr_masks, vr_mask, state):
        # 'it' contains a word index
        # xt = self.embed(it)
        xt = self.word_embedding(self.embed[0].weight, ocr_feats, it)

        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, ocr_feats, p_ocr_feats, ocr_masks, state, att_masks)
        fixed_scores = self.logit(output)  # (b, vocab_size)
        dynamic_ocr_scores = self.ocr_ptr_net(state[0][2], ocr_feats, ocr_masks) # (b, ocr_size)
        dynamic_ocr_scores = self.logit2(dynamic_ocr_scores)
        scores = torch.cat([fixed_scores, dynamic_ocr_scores], dim=-1)

        probs = F.softmax(scores, dim=1)
        
        zeroed_masks = torch.zeros_like(fixed_scores)
        entended_vr_mask = torch.cat((zeroed_masks, vr_mask), dim = -1)
        entended_vr_mask = entended_vr_mask + torch.ones_like(entended_vr_mask)
        probs = probs * entended_vr_mask
        probs = probs/probs.sum(1,keepdim=True)

        logprobs = torch.log(probs)
        fixed_masks = torch.ones_like(fixed_scores)
        both_masks = torch.cat((fixed_masks, ocr_masks), dim = -1)
        both_masks = (1.0 - both_masks) * -10000.0
        logprobs = logprobs + both_masks

        return logprobs, state

    def _sample_beam(self, ocr_feats, ocr_masks, ocr_relations, fc_feats, att_feats, att_masks=None, unk_idx=0, ocrunk_idx=0, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, p_ocr_feats, pp_ocr_feats, p_ocr_masks, vocab_rel_mask = self._prepare_feature(fc_feats, att_feats, att_masks, ocr_feats, ocr_masks, ocr_relations)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = p_fc_feats[k:k+1].expand(beam_size, p_fc_feats.size(1))
            tmp_att_feats = p_att_feats[k:k+1].expand(*((beam_size,)+p_att_feats.size()[1:])).contiguous()
            tmp_p_att_feats = pp_att_feats[k:k+1].expand(*((beam_size,)+pp_att_feats.size()[1:])).contiguous()
            tmp_att_masks = p_att_masks[k:k+1].expand(*((beam_size,)+p_att_masks.size()[1:])).contiguous() if att_masks is not None else None
            tmp_ocr_feats = p_ocr_feats[k:k+1].expand(*((beam_size,)+p_ocr_feats.size()[1:])).contiguous()
            tmp_p_ocr_feats = pp_ocr_feats[k:k+1].expand(*((beam_size,)+pp_ocr_feats.size()[1:])).contiguous()
            tmp_ocr_masks = p_ocr_masks[k:k+1].expand(*((beam_size,)+p_ocr_masks.size()[1:])).contiguous()

            appeared_words = torch.zeros((beam_size, self.vocab_size+1+self.ocr_size)).cuda()

            for t in range(1):
                if t == 0: # input <bos>
                    it = fc_feats.new_zeros([beam_size], dtype=torch.long)

                vr_mask = vocab_rel_mask[k,it[0]].unsqueeze(0)
                tmp_vr_masks = vr_mask.expand(*((beam_size,)+vr_mask.size()[1:])).contiguous()

                logprobs, state = self.get_logprobs_state_test(it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, tmp_ocr_feats, tmp_p_ocr_feats, tmp_ocr_masks, tmp_vr_masks, state)
                logprobs[..., unk_idx] = -1e10
                logprobs[..., ocrunk_idx] = -1e10

            self.done_beams[k] = self.beam_search(self.vocab_size+1, appeared_words, vocab_rel_mask[k], unk_idx, ocrunk_idx, state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, tmp_ocr_feats, tmp_p_ocr_feats, tmp_ocr_masks, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, ocr_feats, ocr_masks, ocr_relations, fc_feats, att_feats, att_masks=None, unk_idx=0, ocrunk_idx=0, opt={}):

        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        if beam_size > 1:
            return self._sample_beam(ocr_feats, ocr_masks, ocr_relations, fc_feats, att_feats, att_masks, unk_idx, ocrunk_idx, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, p_ocr_feats, pp_ocr_feats, p_ocr_masks, vocab_rel_mask = self._prepare_feature(fc_feats, att_feats, att_masks, ocr_feats, ocr_masks, ocr_relations)

        seq = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)

            for bi in range(batch_size):
                if bi == 0:
                    vr_mask = vocab_rel_mask[0,it[0]].unsqueeze(0)
                else:
                    vr_mask = torch.cat([vr_mask, vocab_rel_mask[bi,it[bi]].unsqueeze(0)], dim=0)

            logprobs, state = self.get_logprobs_state_test(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, p_ocr_feats, pp_ocr_feats, p_ocr_masks, vr_mask, state)
            logprobs[..., unk_idx] = -1e10
            logprobs[..., ocrunk_idx] = -1e10
            
            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            # sample the next word
            if t == self.seq_length: # skip if we achieve maximum length
                break
            if sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data) # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:,t] = it
            seqLogprobs[:,t] = sampleLogprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs

class TopDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size) # h^1_t, \hat v
        self.ocr_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size) # h^1_t, \hat v
        self.attention = Attention(opt)
        self.attention2 = Attention2(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, ocr_feats, p_ocr_feats, ocr_masks, state, att_masks=None):
        prev_h_lang = state[0][1]
        prev_h_ocr = state[0][2]
        att_lstm_input = torch.cat([prev_h_lang+prev_h_ocr, torch.mean(att_feats, 1), xt], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        lang_lstm_input = torch.cat([att, h_att], 1)
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????
        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        ocr_att = self.attention2(h_att, ocr_feats, p_ocr_feats, ocr_masks)
        ocr_lstm_input = torch.cat([ocr_att, h_att], 1)
        h_ocr, c_ocr = self.ocr_lstm(ocr_lstm_input, (state[0][2], state[1][2]))

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang, h_ocr]), torch.stack([c_att, c_lang, c_ocr]))

        return output, state

class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        
        att_h = self.h2att(h)                        # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
        dot = att + att_h                                   # batch * att_size * att_hid_size
        dot = F.tanh(dot)                                # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size
        
        weight = F.softmax(dot, dim=1)                             # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True) # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1)) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        return att_res

class Attention2(nn.Module):
    def __init__(self, opt):
        super(Attention2, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        
        att_h = self.h2att(h)                        # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
        dot = att + att_h                                   # batch * att_size * att_hid_size
        dot = F.tanh(dot)                                # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size
        
        weight = F.softmax(dot, dim=1)                             # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight_sum = weight.sum(1, keepdim=True)
            for i in range(weight_sum.size(0)):
                if weight_sum[i] == 0:
                    weight_sum[i] = 1
            weight = weight / weight_sum # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1)) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        return att_res

class TopDownModel(AttModel):
    def __init__(self, opt):
        super(TopDownModel, self).__init__(opt)
        self.num_layers = 3
        self.core = TopDownCore(opt)
