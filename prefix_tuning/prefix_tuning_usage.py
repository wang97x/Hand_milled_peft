#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2025/7/3 9:42
@desc: 
"""
import torch
import torch.nn as nn
import torch.optim as optim

from models.simple_transformer_model import SimpleTransformerModel
from prefix_tuning_model import PrefixTuningModel

# 初始化基座模型
vocab_size = 10000
hidden_size = 512
num_layers = 6
model = SimpleTransformerModel(vocab_size, hidden_size, num_layers)

# 初始化 Prefix Tuning 模型
prefix_model = PrefixTuningModel(model, num_virtual_tokens=20, hidden_size=hidden_size)

# 构造虚拟数据
input_ids = torch.randint(100, 200, (4, 128))  # 假设批量大小为4，序列长度为128
attention_mask = torch.ones_like(input_ids)

# 前向传播
outputs = prefix_model(input_ids, attention_mask)

# 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(prefix_model.parameters(), lr=1e-4)

# 训练循环
prefix_model.train()
for epoch in range(3):  # 训练3个epoch
    optimizer.zero_grad()
    outputs = prefix_model(input_ids, attention_mask)
    loss = loss_fn(outputs.view(-1, vocab_size), input_ids.view(-1))
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
    loss.backward()
    optimizer.step()

# 推理示例
prefix_model.eval()
with torch.no_grad():
    outputs = prefix_model(input_ids, attention_mask)
    print("Inference output shape:", outputs.shape)