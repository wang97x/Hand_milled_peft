#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2025/7/3 9:45
@desc: 
"""
import torch
import torch.nn as nn


class SimpleTransformerModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, nhead=2, batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        embedding = self.embeddings(input_ids)
        transformer_output = self.transformer(embedding)
        return self.fc(transformer_output)