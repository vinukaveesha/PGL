# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import pgl


def linear_init(input_size, output_size, init_type='kaiming_uniform', with_bias=True):
    """Initialize linear layer"""
    linear = nn.Linear(input_size, output_size, bias_attr=with_bias)
    
    if init_type == 'kaiming_uniform':
        nn.initializer.KaimingUniform()(linear.weight)
    elif init_type == 'linear':
        bound = 1.0 / math.sqrt(input_size)
        nn.initializer.Uniform(-bound, bound)(linear.weight)
    else:
        nn.initializer.XavierUniform()(linear.weight)
        
    if with_bias:
        nn.initializer.Constant(0.0)(linear.bias)
        
    return linear


class GATConv(nn.Layer):
    """Graph Attention Network Convolution Layer"""
    
    def __init__(self,
                 input_size,
                 hidden_size,
                 feat_drop=0.0,
                 attn_drop=0.0,
                 num_heads=1,
                 concat=True,
                 activation=None,
                 **kwargs):
        super(GATConv, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.concat = concat
        self.activation = activation
        
        self.linear_proj = linear_init(input_size, hidden_size * num_heads, init_type='linear')
        self.attn_proj = linear_init(hidden_size * 2, 1, init_type='linear')
        
        self.feat_dropout = nn.Dropout(feat_drop)
        self.attn_dropout = nn.Dropout(attn_drop)
        
    def forward(self, graph, feature):
        """Forward pass"""
        num_nodes = feature.shape[0]
        
        # Apply feature dropout
        feature = self.feat_dropout(feature)
        
        # Linear transformation
        h = self.linear_proj(feature)  # [N, hidden_size * num_heads]
        h = paddle.reshape(h, [-1, self.num_heads, self.hidden_size])  # [N, num_heads, hidden_size]
        # Compute attention on edges sorted by destination to satisfy segment ops
        src_nodes, dst_nodes, _ = graph.sorted_edges(sort_by="dst")
        
        # Get source and destination features
        h_src = paddle.gather(h, src_nodes, axis=0)  # [E, num_heads, hidden_size]
        h_dst = paddle.gather(h, dst_nodes, axis=0)  # [E, num_heads, hidden_size]
        
        # Concatenate for attention computation
        h_cat = paddle.concat([h_src, h_dst], axis=-1)  # [E, num_heads, 2*hidden_size]
        
        # Compute attention scores
        e = self.attn_proj(h_cat)  # [E, num_heads, 1]
        e = paddle.squeeze(e, axis=-1)  # [E, num_heads]
        e = F.leaky_relu(e)
        
        # Apply attention dropout
        e = self.attn_dropout(e)
        
        # Message aggregation using segment operations
        e_expanded = paddle.unsqueeze(e, axis=-1)  # [E, num_heads, 1]
        weighted_msg = e_expanded * h_src  # [E, num_heads, hidden]

        # Sum over incoming edges per destination node
        dst = dst_nodes
        sum_msg = pgl.math.segment_sum(
            paddle.reshape(weighted_msg, [weighted_msg.shape[0], -1]), dst
        )  # [max(dst)+1, num_heads*hidden]
        sum_msg = paddle.reshape(sum_msg, [-1, self.num_heads, self.hidden_size])

        sum_e = pgl.math.segment_sum(e, dst)  # [max(dst)+1, num_heads]
        sum_rows = sum_e.shape[0]
        # Pad to all nodes if needed
        if sum_rows < feature.shape[0]:
            pad_len = feature.shape[0] - sum_rows
            pad_msg = paddle.zeros([pad_len, self.num_heads, self.hidden_size], dtype=sum_msg.dtype)
            pad_e = paddle.zeros([pad_len, self.num_heads], dtype=sum_e.dtype)
            sum_msg = paddle.concat([sum_msg, pad_msg], axis=0)
            sum_e = paddle.concat([sum_e, pad_e], axis=0)

        sum_e = paddle.clip(sum_e, min=1e-10)
        out = sum_msg / paddle.unsqueeze(sum_e, axis=-1)
        
        if self.concat:
            out = paddle.reshape(out, [-1, self.num_heads * self.hidden_size])
        else:
            out = paddle.mean(out, axis=1)
            
        if self.activation is not None:
            out = self.activation(out)
            
        return out
