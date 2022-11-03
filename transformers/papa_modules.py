import torch
from torch import nn
import json
import os
from .modeling_utils import ModuleUtilsMixin
import numpy as np
from .activations import ACT2FN

class FreezeExtractPoller(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_across_layers = nn.Linear((config.num_hidden_layers+1) * config.hidden_size, config.hidden_size)
        temp_weights = torch.ones(config.num_hidden_layers+1, config.max_seq_length) / config.max_seq_length
        self.weights_per_layer = nn.Parameter(temp_weights, requires_grad=True)
        self.activation = nn.Tanh()
        self.output_dim = config.hidden_size

    def forward(self, hidden_states, mask=None):
        concat_tensor = None

        for i in range(len(hidden_states)):
            w = self.weights_per_layer[i]
            if mask is not None:
                w = torch.unsqueeze(w * mask, dim=1)
            current_tensor = torch.squeeze(torch.bmm(w, hidden_states[i]))
            if concat_tensor is None:
                concat_tensor = current_tensor
            else:
                concat_tensor = torch.cat([concat_tensor, current_tensor], 1)
        pooled_output = self.dense_across_layers(concat_tensor)

        pooled_output = self.activation(pooled_output)
        return pooled_output

class FreezeExtractPollerTokenClassification(nn.Module):
    def __init__(self, config, mlm=False):
        super().__init__()
        self.dense_across_layers_1 = nn.Linear((config.num_hidden_layers+1) * config.hidden_size, config.hidden_size)
        if mlm:
            self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.bias = nn.Parameter(torch.zeros(config.vocab_size))
            # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
            self.decoder.bias = self.bias
            if isinstance(config.hidden_act, str):
                self.activation = ACT2FN[config.hidden_act]
            else:
                self.activation = config.hidden_act
        else:
            self.decoder = nn.Linear(config.hidden_size, config.num_labels)
            self.activation = nn.Tanh()

    def forward(self, hidden_states):
        concat_tensor = None

        for i in range(len(hidden_states)):
            current_tensor = hidden_states[i]
            if concat_tensor is None:
                concat_tensor = current_tensor
            else:
                concat_tensor = torch.cat([concat_tensor, current_tensor], -1)
        pooled_output = self.dense_across_layers_1(concat_tensor)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.decoder(pooled_output)
        return pooled_output


def get_sorting_heads_dict(sorting_heads_dir, num_heads):
    if num_heads == 0:
        return None

    with open(os.path.join(sorting_heads_dir, 'sorted_heads.json'), "r") as fp:
        heads_sorted = json.load(fp)
    layers_dict = {}

    for layer, head in heads_sorted[:num_heads]:
        if layer not in layers_dict:
            layers_dict[layer] = [head]
        else:
            layers_dict[layer].append(head)
    return layers_dict

def get_att_pattern_from_mask(att_mask):
    batch_size = att_mask.shape[0]
    seq_len = att_mask.shape[-1]
    if list(att_mask.shape) == [batch_size, 1, seq_len, seq_len]:
        return att_mask

    my_attention_mask = att_mask * -1
    max_val = torch.max(my_attention_mask)
    if torch.abs(max_val) > 0:
        my_attention_mask /= torch.max(my_attention_mask)
    my_attention_mask = 1 - my_attention_mask
    my_attention_mask = my_attention_mask[:,0]
    full_pattern = torch.bmm(torch.transpose(my_attention_mask, dim0=-1, dim1=-2), my_attention_mask)
    return torch.unsqueeze(full_pattern, 1)

def mask_and_normalize(att_probs, att_mask):
    cur_attention_pattern = get_att_pattern_from_mask(att_mask)
    new_att_probs = att_probs * cur_attention_pattern
    new_att_probs = torch.nn.functional.normalize(new_att_probs, p=1.0, dim=-1)

    return new_att_probs

def combine_static_and_dynamic_att(att_probs, static_heads, att_mask, static_heads_indexes, num_attention_heads):
    att_probs_shape = att_probs.shape
    batch = att_probs_shape[0]
    seq_len = att_probs_shape[-1]

    new_att_probs = torch.zeros((batch, num_attention_heads, seq_len, seq_len), device=att_probs.device)
    dynamic_heads_indexes = list(set(range(num_attention_heads)) - set(static_heads_indexes))

    new_att_probs[:, dynamic_heads_indexes] = att_probs
    new_att_probs[:, static_heads_indexes] = static_heads

    # apply my masking and normalization over all the heads, this is a bit of
    # an overhead, but looks much simpler:
    new_att_probs = mask_and_normalize(new_att_probs, att_mask)

    return new_att_probs