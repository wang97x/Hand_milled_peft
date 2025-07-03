#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2025/7/3 9:35
@desc: 
"""
import torch
import torch.nn as nn
import torch.optim as optim


class PrefixTuningModel(nn.Module):
    def __init__(self, model, num_virtual_tokens, hidden_size):
        super().__init__()
        self.model = model
        self.num_virtual_tokens = num_virtual_tokens
        self.hidden_size = hidden_size

        # 定义虚拟前缀嵌入层
        self.prefix_embedding = nn.Embedding(num_virtual_tokens, hidden_size)
        # 定义前缀 MLP（可选）
        self.prefix_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, input_ids, attention_mask=None):
        batch_size = input_ids.shape[0]
        # 获取虚拟前缀嵌入
        prefix_tokens = torch.arange(self.num_virtual_tokens, device=input_ids.device).expand(batch_size, -1)
        prefix_embedding = self.prefix_embedding(prefix_tokens)
        prefix_embedding = self.prefix_mlp(prefix_embedding)

        # 将前缀嵌入与输入嵌入拼接
        input_embedding = self.model.embeddings(input_ids)
        combined_embedding = torch.cat([prefix_embedding, input_embedding], dim=1)

        # 更新 attention_mask
        if attention_mask is not None:
            prefix_attention_mask = torch.ones((batch_size, self.num_virtual_tokens), device=attention_mask.device)
            combined_attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)
        else:
            combined_attention_mask = None

        # 将拼接后的嵌入和注意力掩码传递给模型
        outputs = self.model(inputs_embeds=combined_embedding, attention_mask=combined_attention_mask)
        return outputs