import logging
import torch
import torch.nn as nn
from transformers import BertTokenizer
import transformers
from .modeling_bert import BertModel
import math
from torch.nn import functional as F
import json
from torch.nn.functional import gelu, relu, tanh
import numpy as np
from torch import nn
from torchvision.models import resnet50, resnet101, resnet152
import timm
import pdb


class crossAttentionLayer(nn.Module):

    def __init__(self, hidden_size, dropout_p=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.softmax = nn.Softmax(dim=-1)
        self.linear_q2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout_p)
        )
        self.linear_v2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout_p)
        )
        self.linear_k2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout_p)
        )

        self.linear_q = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout_p)
        )
        self.linear_v = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout_p)
        )
        self.linear_k = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout_p)
        )
        self.merge_fr = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(0.3),
            nn.ELU(inplace=True)

        )

        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)

    def forward(self, x, y):
        """
        Args:
            q: [B, L_q, D_q]
            k: [B, L_k, D_k]
            v: [B, L_v, D_v]
        Return: Same shape to q, but in 'v' space, soft knn
        """

        # linear projection
        qx = self.linear_q(x)
        kx = self.linear_k(x)
        vx = self.linear_v(x)
        qy = self.linear_q2(y)
        ky = self.linear_k2(y)
        vy = self.linear_v2(y)

        attention = torch.bmm(qx, kx.transpose(-2, -1))
        scale = vx.size(-1) ** -0.5
        attention = attention * scale
        attention = self.softmax(attention)
        output1 = torch.bmm(attention, vx)

        attention = torch.bmm(qx, ky.transpose(-2, -1))
        scale = vy.size(-1) ** -0.5
        attention = attention * scale
        attention = self.softmax(attention)
        output2 = torch.bmm(attention, vy)

        attention = torch.bmm(qy, kx.transpose(-2, -1))
        scale = vx.size(-1) ** -0.5
        attention = attention * scale
        attention = self.softmax(attention)
        output3 = torch.bmm(attention, vx)

        attention = torch.bmm(ky, qy.transpose(-2, -1))
        scale = vy.size(-1) ** -0.5
        attention = attention * scale
        attention = self.softmax(attention)
        output4 = torch.bmm(attention, vy)

        output1 = self.layer_norm(output1 + output2)
        output2 = self.layer_norm(output3 + output4)

        output = torch.cat([output1, output2], dim=1)
        output = self.merge_fr(output)

        return output



class CAMIM(nn.Module):
    def __init__(self, max_length, pretrain_path, blank_padding=True, mask_entity=False):
        """
        max_length: max length of sentence
        pretrain_path: path of pretrain model
        blank_padding: need padding or not
        mask_entity: mask the entity tokens or not
        """
        super().__init__()
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.mask_entity = mask_entity
        self.hidden_size = 768 * 2

        self.nclass = 300
        self.model_resnet50 = timm.create_model('resnet50', pretrained=True, pretrained_cfg_overlay=dict(
            file='resnet50_a1_0-14fe96d1.pth'))  # get the pre-trained ResNet model for the image
        for param in self.model_resnet50.parameters():
            param.requires_grad = True

        logging.info('Loading BERT pre-trained checkpoint.')
        self.bert = transformers.BertModel.from_pretrained(
            'bert-base-uncased')  # get the pre-trained BERT model for the text
        self.bert2 = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_pic = nn.Linear(2048, self.hidden_size // 2)
        self.linear_final = nn.Linear(self.hidden_size * 2 + self.hidden_size, self.hidden_size)

        # the attention mechanism for fine-grained features
        self.linear_q_fine = nn.Linear(768, self.hidden_size // 2)
        self.linear_k_fine = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)
        self.linear_v_fine = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)

        # the attention mechanism for coarse-grained features
        self.linear_q_coarse = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)
        self.linear_k_coarse = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)
        self.linear_v_coarse = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)

        self.linear_weights = nn.Linear(self.hidden_size * 3, 3)
        self.linear_phrases = nn.Linear(self.hidden_size // 2, self.hidden_size)
        self.linear_extend_pic = nn.Linear(self.hidden_size // 2, self.hidden_size // 2)
        self.dropout_linear = nn.Dropout(0.5)

        self.attention = crossAttentionLayer(768, 0.3)

        self.encoder_conv_sentence = nn.Sequential(
            nn.Linear(in_features=768 * 2, out_features=768)
        )
        self.encoder_conv_word = nn.Sequential(
            nn.Linear(in_features=768, out_features=768 * 2),
            nn.Tanh(),
            nn.Linear(in_features=768 * 2, out_features=768)
        )
        self.encoder_conv_back = nn.Sequential(
            nn.Linear(in_features=768, out_features=768 * 2),
            nn.Tanh(),
            nn.Linear(in_features=768 * 2, out_features=768)
        )
        self.linear_x = nn.Linear(self.hidden_size // 2, self.hidden_size)

    def forward(self,
                token,
                att_mask,
                pos1,
                pos2,
                token_phrase,
                att_mask_phrase,
                image_diff,
                image_ori,
                image_diff_objects,
                image_ori_objects,
                weights,
                caption_input_ids,
                caption_token_type_ids,
                caption_attention_mask
                ):
        """
        Args:
            token: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
            pos1: (B, 1), position of the head entity starter
            pos2: (B, 1), position of the tail entity starter
            image_ori: the original images from the dataset
            image_ori_objects: the visual objects from the original images
            image_diff: the generated images from stable-diffusion
            image_diff_objects: the visual objects from the generated images
            weights: the correlation coefficient between text and two types of images
            caption_input_ids: caption bert need
            caption_token_type_ids: caption bert need
            caption_attention_mask: caption bert need
        """

        # caption encode
        caption_output = self.bert2(
            input_ids=caption_input_ids,
            attention_mask=caption_attention_mask,
            token_type_ids=caption_token_type_ids
        )
        caption_word = caption_output.last_hidden_state
        caption_sentence = caption_word.mean(1)

        # text caption encode
        text_output = self.bert2(token[:, 0:40], attention_mask=att_mask[:, 0:40])
        text_word = text_output.last_hidden_state
        text_sentence = text_word.mean(1)

        # visual encode
        feature_DifImg_FineGrained = self.model_resnet50.forward_features(image_diff)
        feature_OriImg_FineGrained = self.model_resnet50.forward_features(image_ori)
        feature_DifImg_CoarseGrained = self.model_resnet50.forward_features(
            image_diff_objects)
        feature_OriImg_CoarseGrained = self.model_resnet50.forward_features(
            image_ori_objects)

        pic_diff = torch.reshape(feature_DifImg_FineGrained, (-1, 2048, 49))
        pic_diff = torch.transpose(pic_diff, 1, 2)
        pic_diff = torch.reshape(pic_diff, (-1, 49, 2048))
        pic_diff = self.linear_pic(pic_diff)
        pic_diff = torch.cat([pic_diff, text_sentence.unsqueeze(1).repeat(1, pic_diff.size(1), 1)],
                             dim=2)
        pic_diff = self.encoder_conv_sentence(pic_diff)
        text_word_dif = self.encoder_conv_word(text_word)
        pic_diff = self.encoder_conv_back(pic_diff)
        pic_diff = self.attention(pic_diff, text_word_dif)
        pic_diff_ = torch.tanh(torch.sum(pic_diff, dim=1))


        pic_ori = torch.reshape(feature_OriImg_FineGrained, (-1, 2048, 49))
        pic_ori = torch.transpose(pic_ori, 1, 2)
        pic_ori = torch.reshape(pic_ori, (-1, 49, 2048))
        pic_ori = self.linear_pic(pic_ori)
        pic_ori = torch.cat([pic_ori, caption_sentence.unsqueeze(1).repeat(1, pic_ori.size(1), 1)],
                            dim=2)
        pic_ori = self.encoder_conv_sentence(pic_ori)
        caption_word_ori = self.encoder_conv_word(caption_word)
        pic_ori = self.encoder_conv_back(pic_ori)
        pic_ori = self.attention(pic_ori, caption_word_ori)
        pic_ori_ = torch.tanh(torch.sum(pic_ori, dim=1))


        pic_diff_objects = torch.reshape(feature_DifImg_CoarseGrained, (-1, 2048, 49))
        pic_diff_objects = torch.transpose(pic_diff_objects, 1, 2)
        pic_diff_objects = torch.reshape(pic_diff_objects, (-1, 3, 49, 2048))
        r_diff = pic_diff_objects.split(1, dim=1)

        t_diff = []
        for aux_diff in r_diff:
            aux_diff = torch.sum(aux_diff, dim=1)
            aux_diff = self.linear_pic(aux_diff)
            pic_diff_objects = torch.cat([aux_diff, text_sentence.unsqueeze(1).repeat(1, aux_diff.size(1), 1)],
                                         dim=2)
            pic_diff_objects = self.encoder_conv_sentence(pic_diff_objects)
            text_word_obj = self.encoder_conv_word(text_word)
            pic_diff_objects = self.encoder_conv_back(pic_diff_objects)
            pic_diff_objects = self.attention(pic_diff_objects, text_word_obj)
            pic_diff_objects = pic_diff_objects.unsqueeze(1)
            t_diff.append(pic_diff_objects)
        pic_diff_objects = torch.cat(t_diff, dim=1)
        pic_diff_objects = torch.sum(pic_diff_objects, dim=2)
        pic_diff_objects_ = torch.tanh(torch.mean(pic_diff_objects, dim=1))


        pic_ori_objects = torch.reshape(feature_OriImg_CoarseGrained, (-1, 2048, 49))
        pic_ori_objects = torch.transpose(pic_ori_objects, 1, 2)
        pic_ori_objects = torch.reshape(pic_ori_objects, (-1, 3, 49, 2048))
        r = pic_ori_objects.split(1, dim=1)
        t = []
        for aux in r:
            aux = torch.sum(aux, dim=1)
            aux = self.linear_pic(aux)
            pic_ori_objects = torch.cat([aux, caption_sentence.unsqueeze(1).repeat(1, aux.size(1), 1)],
                                        dim=2)
            pic_ori_objects = self.encoder_conv_sentence(pic_ori_objects)
            caption_word_obj = self.encoder_conv_word(caption_word)
            pic_ori_objects = self.encoder_conv_back(pic_ori_objects)
            pic_ori_objects = self.attention(pic_ori_objects, caption_word_obj)
            pic_ori_objects = pic_ori_objects.unsqueeze(1)
            t.append(pic_ori_objects)
        pic_ori_objects = torch.cat(t, dim=1)
        pic_ori_objects = torch.sum(pic_ori_objects, dim=2)
        pic_ori_objects_ = torch.tanh(torch.mean(pic_ori_objects, dim=1))

        output_text = self.bert(token, attention_mask=att_mask)
        hidden_text = output_text[0]

        output_phrases = self.bert(token_phrase, attention_mask=att_mask_phrase)
        hidden_phrases = output_phrases[0]

        hidden_k_text = self.linear_k_fine(hidden_text)
        hidden_v_text = self.linear_v_fine(hidden_text)
        pic_q_diff = self.linear_q_fine(pic_diff)
        pic_diffusion = torch.sum(torch.tanh(self.att(pic_q_diff, hidden_k_text, hidden_v_text)), dim=1)

        hidden_k_text = self.linear_k_fine(hidden_text)
        hidden_v_text = self.linear_v_fine(hidden_text)
        pic_q_origin = self.linear_q_fine(pic_ori)
        pic_original = torch.sum(torch.tanh(self.att(pic_q_origin, hidden_k_text, hidden_v_text)), dim=1)  # bsz,768

        hidden_k_phrases = self.linear_k_coarse(hidden_phrases)
        hidden_v_phrases = self.linear_v_coarse(hidden_phrases)
        pic_q_diff_objects = self.linear_q_coarse(pic_diff_objects)
        pic_diffusion_objects = torch.sum(torch.tanh(self.att(pic_q_diff_objects, hidden_k_phrases, hidden_v_phrases)),
                                          dim=1)

        hidden_k_phrases = self.linear_k_coarse(hidden_phrases)
        hidden_v_phrases = self.linear_v_coarse(hidden_phrases)
        pic_q_ori_objects = self.linear_q_coarse(pic_ori_objects)
        pic_original_objects = torch.sum(torch.tanh(self.att(pic_q_ori_objects, hidden_k_phrases, hidden_v_phrases)),
                                         dim=1)

        # coarse-grained textual features
        hidden_phrases = torch.sum(hidden_phrases, dim=1)
        hidden_phrases = self.linear_phrases(hidden_phrases)

        # Get entity start hidden state
        onehot_head = torch.zeros(hidden_text.size()[:2]).float().to(hidden_text.device)  # (B, L)
        onehot_tail = torch.zeros(hidden_text.size()[:2]).float().to(hidden_text.device)  # (B, L)
        onehot_head = onehot_head.scatter_(1, pos1, 1)
        onehot_tail = onehot_tail.scatter_(1, pos2, 1)
        head_hidden = (onehot_head.unsqueeze(2) * hidden_text).sum(1)  # (B, H)
        tail_hidden = (onehot_tail.unsqueeze(2) * hidden_text).sum(1)  # (B, H)
        # fine-grained textual features
        x = torch.cat([head_hidden, tail_hidden], dim=-1)

        pic_ori_final = (pic_original + pic_ori_) * weights[:, 1].reshape(-1, 1) + (
                    pic_original_objects + pic_ori_objects_) * weights[:, 0].reshape(-1, 1)
        pic_diff_final = (pic_diffusion + pic_diff_) * weights[:, 3].reshape(-1, 1) + (
                    pic_diffusion_objects + pic_diff_objects_) * weights[:, 2].reshape(-1, 1)

        pic_ori = torch.tanh(self.linear_extend_pic(pic_ori_final))  # 32,768
        pic_diff = torch.tanh(self.linear_extend_pic(pic_diff_final))  # 32,768

        x = torch.cat([x, hidden_phrases, pic_ori, pic_diff], dim=-1)
        x = self.linear_final(self.dropout_linear(x))  # 32,2*768

        # MI
        x_pred1 = pic_ori
        y_pred1 = pic_diff
        x_pred1 = x_pred1 / x_pred1.norm(dim=1, keepdim=True)
        y_pred1 = y_pred1 / y_pred1.norm(dim=1, keepdim=True)
        pos1 = torch.sum(y_pred1 * x_pred1, dim=-1)  # bsz
        neg1 = torch.logsumexp(torch.matmul(y_pred1, x_pred1.t()), dim=-1)  # bsz
        nce = -(pos1 - neg1).mean()
        

        return x, nce

    def tokenize(self, item):
        """
        Args:
            item: data instance containing 'text' / 'token', 'h' and 't'
        Return:
            Name of the relation of the sentence
        """
        if len(item) == 5 and not isinstance(item, dict):
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(item)
            indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)

            return indexed_tokens

        # Sentence -> token
        if 'text' in item:
            sentence = item['text']
            is_token = False
        elif 'token' in item:
            sentence = item['token']
            is_token = True

        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        pos_min = pos_head
        pos_max = pos_tail
        if pos_head[0] > pos_tail[0]:
            pos_min = pos_tail
            pos_max = pos_head
            rev = True
        else:
            rev = False

        if not is_token:
            sent0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
            ent0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
            sent1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
            ent1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
            sent2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
        else:
            sent0 = self.tokenizer.tokenize(' '.join(sentence[:pos_min[0]]))
            ent0 = self.tokenizer.tokenize(' '.join(sentence[pos_min[0]:pos_min[1]]))
            sent1 = self.tokenizer.tokenize(' '.join(sentence[pos_min[1]:pos_max[0]]))
            ent1 = self.tokenizer.tokenize(' '.join(sentence[pos_max[0]:pos_max[1]]))
            sent2 = self.tokenizer.tokenize(' '.join(sentence[pos_max[1]:]))

        if self.mask_entity:
            ent0 = ['[unused4]'] if not rev else ['[unused5]']
            ent1 = ['[unused5]'] if not rev else ['[unused4]']
        else:
            ent0 = ['[unused0]'] + ent0 + ['[unused1]'] if not rev else ['[unused2]'] + ent0 + ['[unused3]']
            ent1 = ['[unused2]'] + ent1 + ['[unused3]'] if not rev else ['[unused0]'] + ent1 + ['[unused1]']

        re_tokens = ['[CLS]'] + sent0 + ent0 + sent1 + ent1 + sent2 + ['[SEP]']
        pos1 = 1 + len(sent0) if not rev else 1 + len(sent0 + ent0 + sent1)
        pos2 = 1 + len(sent0 + ent0 + sent1) if not rev else 1 + len(sent0)
        pos1 = min(self.max_length - 1, pos1)
        pos2 = min(self.max_length - 1, pos2)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)

        # Position
        pos1 = torch.tensor([[pos1]]).long()
        pos2 = torch.tensor([[pos2]]).long()

        # Padding
        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1

        phrases = item['grounding']
        token_phrases = self.tokenizer.convert_tokens_to_ids(phrases.split(' '))
        while len(token_phrases) < 6:
            token_phrases.append(0)
        token_phrases = token_phrases[:6]
        token_phrases = torch.tensor(token_phrases).long().unsqueeze(0)
        att_mask_phrases = torch.zeros(token_phrases.size()).long()
        return indexed_tokens, att_mask, pos1, pos2, token_phrases, att_mask_phrases

    # the attention mechanism
    def att(self, query, key, value):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)  # (5,50)
        att_map = F.softmax(scores, dim=-1)

        return torch.matmul(att_map, value)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.05)


class MLP(nn.Module):
    def __init__(self, input_sizes, dropout_prob=0.2, bias=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(1, len(input_sizes)):
            self.layers.append(nn.Linear(input_sizes[i - 1], input_sizes[i], bias=bias))
        self.norm_layers = nn.ModuleList()
        if len(input_sizes) > 2:
            for i in range(1, len(input_sizes) - 1):
                self.norm_layers.append(nn.LayerNorm(input_sizes[i]))
        self.drop_out = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(self.drop_out(x))
            if i < len(self.layers) - 1:
                # x = gelu(x)
                x = relu(x)
                if len(self.norm_layers):
                    x = self.norm_layers[i](x)
        return x




