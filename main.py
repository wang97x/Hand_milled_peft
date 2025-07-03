#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2025/7/3 9:44
@desc: 
"""
import argparse

import torch


def run_prefix_tuning():
    """运行 Prefix Tuning 示例"""
    from prefix_tuning.prefix_tuning_usage import prefix_model, input_ids, attention_mask, vocab_size
    from torch.nn import CrossEntropyLoss
    from torch.optim import Adam

    # 定义损失函数和优化器
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(prefix_model.parameters(), lr=1e-4)

    # 训练循环
    prefix_model.train()
    for epoch in range(3):  # 训练3个epoch
        optimizer.zero_grad()
        outputs = prefix_model(input_ids, attention_mask)
        loss = loss_fn(outputs.view(-1, vocab_size), input_ids.view(-1))
        print(f"Prefix Tuning - Epoch {epoch + 1}, Loss: {loss.item()}")
        loss.backward()
        optimizer.step()

    # 推理示例
    prefix_model.eval()
    with torch.no_grad():
        outputs = prefix_model(input_ids, attention_mask)
        print("Prefix Tuning - Inference output shape:", outputs.shape)

def run_lora():
    """运行 LoRA 示例"""
    from lora.lora_usage import lora_model, input_ids, attention_mask, vocab_size
    from torch.nn import CrossEntropyLoss
    from torch.optim import Adam

    # 定义损失函数和优化器
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(lora_model.parameters(), lr=1e-4)

    # 训练循环
    lora_model.train()
    for epoch in range(3):  # 训练3个epoch
        optimizer.zero_grad()
        outputs = lora_model(input_ids, attention_mask)
        loss = loss_fn(outputs.view(-1, vocab_size), input_ids.view(-1))
        print(f"LoRA - Epoch {epoch + 1}, Loss: {loss.item()}")
        loss.backward()
        optimizer.step()

    # 推理示例
    lora_model.eval()
    with torch.no_grad():
        outputs = lora_model(input_ids, attention_mask)
        print("LoRA - Inference output shape:", outputs.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PEFT example.")
    parser.add_argument("method", type=str, choices=["prefix_tuning", "lora"], help="Choose to run Prefix Tuning or LoRA")
    args = parser.parse_args()

    if args.method == "prefix_tuning":
        run_prefix_tuning()
    elif args.method == "lora":
        run_lora()