#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2025/7/3 9:42
@desc: 
"""
import torch
import torch.nn as nn  # 导入基座模型


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, r, alpha, bias=True):
        super().__init__()
        self.r = r
        self.alpha = alpha

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.lora_A = nn.Parameter(torch.randn(r, in_features))
        self.lora_B = nn.Parameter(torch.randn(out_features, r))
        self.weight.requires_grad = False

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        lora_matrix = self.lora_B @ self.lora_A
        scaled_lora_matrix = (self.alpha / self.r) * lora_matrix
        output = x @ (self.weight.T + scaled_lora_matrix.T)
        if self.bias is not None:
            output += self.bias
        return output


class LoRAModel(nn.Module):
    def __init__(self, model, r, alpha):
        super().__init__()
        self.model = model
        self.r = r
        self.alpha = alpha

        # 先收集需要替换的线性层
        replace_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                replace_layers.append((name, module))

        # 再进行替换 ##在替换 nn.Linear 时，需传递原始 bias 参数和数值：
        for name, module in replace_layers:
            parent = self.model
            name_parts = name.split('.')
            for part in name_parts[:-1]:
                parent = getattr(parent, part)
            lora_layer = LoRALayer(
                module.in_features,
                module.out_features,
                self.r,
                self.alpha,
                bias=module.bias is not None
            )
            if module.bias is not None:
                lora_layer.bias.data = module.bias.data.clone()
            setattr(parent, name_parts[-1], lora_layer)

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask)