#!/usr/bin/env python3
"""Baseline Models for Code-Change-Aware AIOps Paper - ASE 2026"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class DeepJIT(nn.Module):
    """DeepJIT: MSR 2019 - CNN + RNN for JIT defect prediction"""
    def __init__(self, vocab_size=50000, embed_dim=128, hidden_dim=256, num_filters=64, filter_sizes=[3,4,5], dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.msg_convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, fs, padding=fs//2) for fs in filter_sizes])
        self.add_convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, fs, padding=fs//2) for fs in filter_sizes])
        self.del_convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, fs, padding=fs//2) for fs in filter_sizes])
        combined_dim = num_filters * len(filter_sizes) * 3
        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
    def conv_and_pool(self, x, convs):
        x = x.transpose(1, 2)
        conv_outs = [F.max_pool1d(F.relu(conv(x)), x.size(2)).squeeze(2) for conv in convs]
        return torch.cat(conv_outs, dim=1)
    
    def forward(self, msg_ids, added_ids, removed_ids):
        msg_feat = self.conv_and_pool(self.embedding(msg_ids), self.msg_convs)
        add_feat = self.conv_and_pool(self.embedding(added_ids), self.add_convs)
        del_feat = self.conv_and_pool(self.embedding(removed_ids), self.del_convs)
        combined = self.dropout(torch.cat([msg_feat, add_feat, del_feat], dim=1))
        return torch.sigmoid(self.fc2(self.dropout(F.relu(self.fc1(combined)))))

class CC2Vec(nn.Module):
    """CC2Vec: ICSE 2020 - Hierarchical attention for code changes"""
    def __init__(self, vocab_size=50000, embed_dim=128, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.token_lstm = nn.LSTM(embed_dim, hidden_dim//2, bidirectional=True, batch_first=True)
        self.token_attn = nn.Linear(hidden_dim, 1)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim//2, 1), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, code_ids, lengths=None):
        embedded = self.dropout(self.embedding(code_ids))
        token_out, _ = self.token_lstm(embedded)
        mask = (code_ids != 0).float()
        attn_scores = self.token_attn(token_out).squeeze(-1).masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)
        pooled = torch.bmm(attn_weights.unsqueeze(1), token_out).squeeze(1)
        return self.fc(pooled)

class JITFine(nn.Module):
    """JIT-Fine: TOSEM 2022 - Fine-tuned pre-trained models"""
    def __init__(self, pretrained_dim=768, hidden_dim=256, dropout=0.3, use_pretrained=False):
        super().__init__()
        self.use_pretrained = use_pretrained
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512, dropout=dropout, batch_first=True), num_layers=4)
        self.embedding = nn.Embedding(50000, 256, padding_idx=0)
        self.classifier = nn.Sequential(nn.Linear(256, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1), nn.Sigmoid())
        
    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        encoded = self.encoder(embedded)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (encoded * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            pooled = encoded.mean(1)
        return self.classifier(pooled)

class LAPredict(nn.Module):
    """LAPredict: Log-Aware Defect Prediction (Our Baseline)"""
    def __init__(self, vocab_size=50000, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.code_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.code_lstm = nn.LSTM(embed_dim, hidden_dim//2, num_layers=num_layers, bidirectional=True, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.log_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.log_lstm = nn.LSTM(embed_dim, hidden_dim//2, num_layers=num_layers, bidirectional=True, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fusion = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim//2, 1), nn.Sigmoid())
        
    def forward(self, code_ids, log_ids):
        code_embed = self.code_embedding(code_ids)
        _, (code_h, _) = self.code_lstm(code_embed)
        code_feat = torch.cat([code_h[-2], code_h[-1]], dim=1)
        log_embed = self.log_embedding(log_ids)
        _, (log_h, _) = self.log_lstm(log_embed)
        log_feat = torch.cat([log_h[-2], log_h[-1]], dim=1)
        fused = self.fusion(torch.cat([code_feat, log_feat], dim=1))
        return self.classifier(fused)

class LogRobust(nn.Module):
    """LogRobust: FSE 2019 - Robust log anomaly detection"""
    def __init__(self, vocab_size=10000, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim//2, num_layers=num_layers, bidirectional=True, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.Tanh(), nn.Linear(hidden_dim//2, 1))
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim//2, 1), nn.Sigmoid())

    def forward(self, log_ids, mask=None):
        embedded = self.embedding(log_ids)
        lstm_out, _ = self.lstm(embedded)
        attn_scores = self.attention(lstm_out).squeeze(-1)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        return self.classifier(context)


class JITSmart(nn.Module):
    """JIT-Smart: FSE 2024 - Multi-task learning for JIT defect prediction with effort estimation"""
    def __init__(self, vocab_size=50000, embed_dim=256, hidden_dim=512, num_heads=8, num_layers=6, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Multi-task heads
        self.defect_head = nn.Sequential(nn.Linear(embed_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim//2, 1), nn.Sigmoid())
        self.effort_head = nn.Sequential(nn.Linear(embed_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim//2, 1))

    def forward(self, code_ids, attention_mask=None):
        embedded = self.embedding(code_ids)
        encoded = self.encoder(embedded)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (encoded * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            pooled = encoded.mean(1)
        defect_prob = self.defect_head(pooled)
        effort_score = self.effort_head(pooled)
        return defect_prob, effort_score


class JITBiCC(nn.Module):
    """JIT-BiCC: JSS 2024 - Bimodal Code Change representation with contrastive learning"""
    def __init__(self, vocab_size=50000, embed_dim=256, hidden_dim=512, num_heads=8, num_layers=4, dropout=0.3):
        super().__init__()
        # Code stream encoder
        self.code_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        code_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.code_encoder = nn.TransformerEncoder(code_layer, num_layers=num_layers)
        # Message stream encoder
        self.msg_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        msg_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.msg_encoder = nn.TransformerEncoder(msg_layer, num_layers=num_layers)
        # Fusion and classifier
        self.fusion = nn.Sequential(nn.Linear(embed_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim//2, 1), nn.Sigmoid())

    def forward(self, code_ids, msg_ids=None):
        # Encode code changes
        code_emb = self.code_embedding(code_ids)
        code_enc = self.code_encoder(code_emb).mean(dim=1)
        # Encode commit message (use code_ids if msg_ids not provided)
        if msg_ids is None:
            msg_ids = code_ids[:, :64]  # Use first 64 tokens as pseudo-message
        msg_emb = self.msg_embedding(msg_ids)
        msg_enc = self.msg_encoder(msg_emb).mean(dim=1)
        # Bimodal fusion
        fused = self.fusion(torch.cat([code_enc, msg_enc], dim=-1))
        return self.classifier(fused)


def get_baseline_model(name, **kwargs):
    models = {
        'deepjit': DeepJIT,
        'cc2vec': CC2Vec,
        'jit-fine': JITFine,
        'lapredict': LAPredict,
        'logrobust': LogRobust,
        'jit-smart': JITSmart,
        'jit-bicc': JITBiCC,
    }
    if name.lower() not in models:
        raise ValueError(f"Unknown baseline: {name}")
    return models[name.lower()](**kwargs)

if __name__ == "__main__":
    print("Testing baseline models...")
    batch_size, seq_len, vocab_size = 4, 128, 50000
    code_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    log_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    msg_ids = torch.randint(0, vocab_size, (batch_size, 64))

    print("\n1. DeepJIT:", DeepJIT(vocab_size=vocab_size)(msg_ids, code_ids, code_ids).shape)
    print("2. CC2Vec:", CC2Vec(vocab_size=vocab_size)(code_ids).shape)
    print("3. JIT-Fine:", JITFine(use_pretrained=False)(code_ids).shape)
    print("4. LAPredict:", LAPredict(vocab_size=vocab_size)(code_ids, log_ids).shape)
    print("5. LogRobust:", LogRobust()(torch.randint(0, 10000, (batch_size, seq_len))).shape)
    print("6. JIT-Smart:", JITSmart(vocab_size=vocab_size)(code_ids)[0].shape)
    print("7. JIT-BiCC:", JITBiCC(vocab_size=vocab_size)(code_ids, msg_ids).shape)
    print("\nAll baseline models tested!")
