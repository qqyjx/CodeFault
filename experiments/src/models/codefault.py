"""
CodeFault: Code-Change-Aware Fault Prediction Model

This module implements the main CodeFault model architecture:
- Code Change Encoder (CodeT5+)
- Fault Context Encoder (Temporal Transformer)
- Cross-Modal Fusion
- Risk Prediction Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class CodeChangeEncoder(nn.Module):
    """Encodes code changes using CodeT5+."""
    
    def __init__(self, model_name: str = "Salesforce/codet5p-770m", hidden_size: int = 768):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.hidden_size = hidden_size
        
        # Change-Aware Attention
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Args:
            input_ids: Tokenized code diff [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            code_embedding: Pooled code embedding [batch, hidden_size]
            hidden_states: Full token representations [batch, seq_len, hidden_size]
            attn_weights: Change-aware attention weights [batch, seq_len]
        """
        # Get CodeT5+ hidden states
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]

        # Change-Aware Attention
        cls_token = hidden_states[:, 0, :]  # [batch, hidden]

        Q = self.query_proj(hidden_states)  # [batch, seq_len, hidden]
        K = self.key_proj(cls_token).unsqueeze(1)  # [batch, 1, hidden]
        V = self.value_proj(hidden_states)  # [batch, seq_len, hidden]

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        scores = scores.squeeze(-1)  # [batch, seq_len]
        scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)

        # Weighted sum
        code_embedding = torch.sum(attn_weights.unsqueeze(-1) * V, dim=1)

        return code_embedding, hidden_states, attn_weights


class FaultContextEncoder(nn.Module):
    """Encodes system logs using Temporal Transformer."""
    
    def __init__(self, vocab_size: int = 10000, hidden_size: int = 768, 
                 num_layers: int = 4, num_heads: int = 8, max_seq_len: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Anomaly-Aware Pooling
        self.anomaly_scorer = nn.Linear(hidden_size, 1)

        # Temporal Decay (Eq. 13): w_i = exp(-beta * delta_t_i / delta_t_max)
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, log_ids: torch.Tensor, attention_mask: torch.Tensor,
                timestamps: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            log_ids: Tokenized log templates [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            timestamps: Relative timestamps [batch, seq_len] (0-1 normalized),
                       if None, linear decay from 1.0 to 0.0 is assumed

        Returns:
            context_embedding: Pooled log context embedding [batch, hidden_size]
            hidden_states: Full token representations [batch, seq_len, hidden_size]
            anomaly_scores: Per-token anomaly scores [batch, seq_len]
        """
        batch_size, seq_len = log_ids.shape

        # Embeddings
        positions = torch.arange(seq_len, device=log_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.embedding(log_ids) + self.pos_embedding(positions)

        # Transformer encoding
        src_key_padding_mask = attention_mask == 0
        hidden_states = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Anomaly-Aware Pooling
        anomaly_scores = torch.sigmoid(self.anomaly_scorer(hidden_states)).squeeze(-1)
        anomaly_scores = anomaly_scores.masked_fill(attention_mask == 0, 0.0)

        # Temporal Decay weights
        if timestamps is not None:
            decay_weights = torch.exp(-self.beta * timestamps)
        else:
            # Default: linear time proxy (later tokens = more recent = higher weight)
            t = torch.linspace(1.0, 0.0, seq_len, device=log_ids.device)
            t = t.unsqueeze(0).expand(batch_size, -1)
            decay_weights = torch.exp(-self.beta * t)
        decay_weights = decay_weights.masked_fill(attention_mask == 0, 0.0)

        # Combined weights: anomaly score * temporal decay
        combined_weights = anomaly_scores * decay_weights

        # Weighted sum
        context_embedding = torch.sum(combined_weights.unsqueeze(-1) * hidden_states, dim=1)
        context_embedding = context_embedding / (combined_weights.sum(dim=1, keepdim=True) + 1e-8)

        return context_embedding, hidden_states, anomaly_scores


class CrossModalFusion(nn.Module):
    """Fuses code and log embeddings through cross-attention."""

    def __init__(self, hidden_size: int = 768, num_heads: int = 8):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(0.1)
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, code_emb: torch.Tensor, context_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            code_emb: Code change embedding [batch, hidden]
            context_emb: Log context embedding [batch, hidden]

        Returns:
            Fused embedding [batch, hidden]
        """
        # Add sequence dimension
        code_emb = code_emb.unsqueeze(1)  # [batch, 1, hidden]
        context_emb = context_emb.unsqueeze(1)  # [batch, 1, hidden]

        # Cross-attention: code attends to context
        attn_output, _ = self.cross_attention(
            query=code_emb,
            key=context_emb,
            value=context_emb
        )

        # Residual + LayerNorm
        x = self.layer_norm(code_emb + attn_output)

        # FFN
        x = self.layer_norm2(x + self.ffn(x))

        return x.squeeze(1)


class FaultSpectrumAlignment(nn.Module):
    """Hierarchical Fault Spectrum Alignment / HFSA (Section 3.3).

    Paper name: Hierarchical Fault Spectrum Alignment (HFSA).
    Learns a shared K-way fault type distribution from both code and log
    embeddings, then aligns them via KL divergence. In the paper, this is
    extended to a two-level hierarchy (K_c=4 coarse + K_f=8 fine); this
    implementation uses a flat K for simplicity.
    """

    def __init__(self, hidden_size: int = 768, num_fault_types: int = 9):
        super().__init__()
        self.num_fault_types = num_fault_types
        # Code spectrum predictor: P_c(T|c) = softmax(W_t * h_c + b_t)
        self.code_predictor = nn.Linear(hidden_size, num_fault_types)
        # Log evidence estimator: P_r(T|r) = softmax(W_e * h_r + b_e)
        self.log_predictor = nn.Linear(hidden_size, num_fault_types)

    def forward(self, code_emb: torch.Tensor, log_emb: torch.Tensor):
        """
        Args:
            code_emb: Code change embedding [batch, hidden]
            log_emb: Log context embedding [batch, hidden]

        Returns:
            P_c: Code-side fault type distribution [batch, K]
            P_r: Log-side fault type distribution [batch, K]
        """
        P_c = F.softmax(self.code_predictor(code_emb), dim=-1)
        P_r = F.softmax(self.log_predictor(log_emb), dim=-1)
        return P_c, P_r

    def alignment_loss(self, P_c: torch.Tensor, P_r: torch.Tensor,
                       labels: torch.Tensor, fault_types: torch.Tensor = None) -> torch.Tensor:
        """Compute spectrum alignment loss.

        For fault-inducing pairs: KL(P_r || P_c) — align log evidence to code prediction.
        For benign pairs: CE(P_c, t_normal) — push code prediction toward normal class.

        Args:
            P_c: Code-side distribution [batch, K]
            P_r: Log-side distribution [batch, K]
            labels: Binary fault labels [batch] (1=fault, 0=benign)
            fault_types: Ground truth fault type indices [batch] (optional)

        Returns:
            Combined alignment loss scalar
        """
        fault_mask = labels.bool()
        benign_mask = ~fault_mask

        loss = torch.tensor(0.0, device=P_c.device)

        # Fault pairs: KL divergence alignment
        if fault_mask.any():
            P_c_fault = P_c[fault_mask]
            P_r_fault = P_r[fault_mask]
            # KL(P_r || P_c) = sum(P_r * log(P_r / P_c))
            kl = F.kl_div(P_c_fault.log(), P_r_fault, reduction='batchmean')
            loss = loss + kl

            # Optional: supervised fault type classification.
            # Fix (2026-06-19): the previous version invoked
            #   self.code_predictor(P_c_fault[valid])
            # but P_c_fault is already a probability distribution of shape
            # (B', num_fault_types) produced by code_predictor + softmax above,
            # so re-applying the Linear(hidden_size -> num_fault_types) layer
            # produces a shape mismatch (B', num_fault_types) x (hidden_size,
            # num_fault_types). Use NLL directly on log-P_c so the supervised
            # term acts on the same predictions already used for KL alignment.
            if fault_types is not None:
                ft_fault = fault_types[fault_mask]
                valid = ft_fault >= 0
                if valid.any():
                    log_p = P_c_fault[valid].clamp(min=1e-10).log()
                    loss = loss + F.nll_loss(log_p, ft_fault[valid]) * 0.5

        # Benign pairs: push toward normal class (last index)
        if benign_mask.any():
            normal_idx = self.num_fault_types - 1
            target = torch.full((benign_mask.sum(),), normal_idx,
                              device=P_c.device, dtype=torch.long)
            loss = loss + F.cross_entropy(P_c[benign_mask], target)

        return loss


class TemporalCausalFusion(nn.Module):
    """Spectrum-Guided Cross-Modal Attention / SGCA (Section 3.4).

    Paper name: Spectrum-Guided Cross-Modal Attention (SGCA).
    Performs true token-level cross-attention between code and log token
    sequences, with spectrum agreement injected as an additive attention
    bias. Pools AFTER attention to produce the fused embedding.
    """

    def __init__(self, hidden_size: int = 768, num_heads: int = 8,
                 num_fault_types: int = 9):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Spectrum bias: project spectrum agreement to per-head scalar bias
        self.spectrum_bias_proj = nn.Linear(num_fault_types, num_heads)

        # Spectrum gate: g = sigma(w_g^T (P_c * P_r) + b_g)
        self.gate = nn.Linear(num_fault_types, 1)

        # Forward attention (code → log): code queries, log keys/values
        self.forward_q = nn.Linear(hidden_size, hidden_size)
        self.forward_k = nn.Linear(hidden_size, hidden_size)
        self.forward_v = nn.Linear(hidden_size, hidden_size)
        self.forward_out = nn.Linear(hidden_size, hidden_size)

        # Backward attention (log → code): log queries, code keys/values
        self.backward_q = nn.Linear(hidden_size, hidden_size)
        self.backward_k = nn.Linear(hidden_size, hidden_size)
        self.backward_v = nn.Linear(hidden_size, hidden_size)
        self.backward_out = nn.Linear(hidden_size, hidden_size)

        # Learnable causal direction weight, initialized to 0.7
        self.gamma = nn.Parameter(torch.tensor(0.7))

        self.attn_dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(0.1)
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def _spectrum_biased_attention(self, Q_proj, K_proj, V_proj, out_proj,
                                    query_seq, kv_seq, spectrum_bias,
                                    kv_mask=None):
        """Token-level multi-head attention with additive spectrum bias.

        Args:
            Q_proj, K_proj, V_proj, out_proj: Linear projections
            query_seq: [batch, q_len, hidden]
            kv_seq: [batch, kv_len, hidden]
            spectrum_bias: [batch, num_heads] additive bias per head
            kv_mask: [batch, kv_len] key padding mask (1=valid, 0=pad)

        Returns:
            attended: [batch, q_len, hidden]
        """
        B, Q_len, _ = query_seq.shape
        _, KV_len, _ = kv_seq.shape
        H = self.num_heads
        D = self.hidden_size // H

        Q = Q_proj(query_seq).view(B, Q_len, H, D).transpose(1, 2)  # [B, H, Q, D]
        K = K_proj(kv_seq).view(B, KV_len, H, D).transpose(1, 2)    # [B, H, KV, D]
        V = V_proj(kv_seq).view(B, KV_len, H, D).transpose(1, 2)    # [B, H, KV, D]

        # Scaled dot-product attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)  # [B, H, Q, KV]

        # Add spectrum agreement bias (broadcast across Q and KV positions)
        bias = spectrum_bias.unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
        scores = scores + bias

        # Apply key padding mask
        if kv_mask is not None:
            mask = (kv_mask == 0).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, KV]
            scores = scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attended = torch.matmul(attn_weights, V)  # [B, H, Q, D]
        attended = attended.transpose(1, 2).contiguous().view(B, Q_len, -1)
        return out_proj(attended)

    def forward(self, code_emb: torch.Tensor, log_emb: torch.Tensor,
                P_c: torch.Tensor, P_r: torch.Tensor,
                code_hidden: torch.Tensor = None, log_hidden: torch.Tensor = None,
                code_mask: torch.Tensor = None, log_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            code_emb: Pooled code embedding [batch, hidden] (used for residual)
            log_emb: Pooled log embedding [batch, hidden] (unused, kept for API compat)
            P_c: Code spectrum distribution [batch, K]
            P_r: Log spectrum distribution [batch, K]
            code_hidden: Code token sequences [batch, code_len, hidden] (token-level)
            log_hidden: Log token sequences [batch, log_len, hidden] (token-level)
            code_mask: Code attention mask [batch, code_len]
            log_mask: Log attention mask [batch, log_len]

        Returns:
            Fused embedding [batch, hidden]
        """
        # Fallback for backward compatibility: if no hidden states, use pooled
        if code_hidden is None:
            code_hidden = code_emb.unsqueeze(1)
        if log_hidden is None:
            log_hidden = log_emb.unsqueeze(1)

        # Spectrum agreement → per-head attention bias
        agreement = P_c * P_r  # [batch, K]
        spectrum_bias = self.spectrum_bias_proj(agreement)  # [batch, num_heads]

        # Spectrum gate (post-attention modulation)
        g = torch.sigmoid(self.gate(agreement))  # [batch, 1]

        # Forward: code tokens attend to log tokens (code → log causality)
        f_cr = self._spectrum_biased_attention(
            self.forward_q, self.forward_k, self.forward_v, self.forward_out,
            code_hidden, log_hidden, spectrum_bias, kv_mask=log_mask
        )  # [batch, code_len, hidden]

        # Backward: log tokens attend to code tokens (log → code causality)
        f_rc = self._spectrum_biased_attention(
            self.backward_q, self.backward_k, self.backward_v, self.backward_out,
            log_hidden, code_hidden, spectrum_bias, kv_mask=code_mask
        )  # [batch, log_len, hidden]

        # Pool after attention (mean pooling over valid tokens)
        if code_mask is not None:
            code_weights = code_mask.float().unsqueeze(-1)  # [B, code_len, 1]
            f_cr_pooled = (f_cr * code_weights).sum(1) / (code_weights.sum(1) + 1e-8)
        else:
            f_cr_pooled = f_cr.mean(dim=1)

        if log_mask is not None:
            log_weights = log_mask.float().unsqueeze(-1)
            f_rc_pooled = (f_rc * log_weights).sum(1) / (log_weights.sum(1) + 1e-8)
        else:
            f_rc_pooled = f_rc.mean(dim=1)

        # Directional fusion with learnable gamma
        gamma = torch.sigmoid(self.gamma)
        fused = g * (gamma * f_cr_pooled + (1 - gamma) * f_rc_pooled)

        # Residual + LayerNorm + FFN
        fused = self.layer_norm(code_emb + fused)
        fused = self.layer_norm2(fused + self.ffn(fused))

        return fused


class CodeFault(nn.Module):
    """
    CodeFault: Complete model for Code-Change-Aware Fault Prediction.

    Architecture (matching paper Sections 3.1-3.5):
    - Code Change Encoder: CodeT5+ with Change-Aware Attention
    - Fault Context Encoder: Temporal Transformer with Anomaly-Aware Pooling + Temporal Decay
    - Fault Spectrum Alignment: K-way fault type distribution alignment (KL + CE)
    - Temporal Causal Fusion: Spectrum-guided bidirectional cross-modal attention
    - Contrastive Loss: In-batch negatives with temperature scaling

    Training loss: L = L_BCE + lambda_1 * L_contrastive + lambda_2 * L_align + lambda_3 * L_normal
    """

    def __init__(self, config: dict = None):
        super().__init__()
        config = config or {}

        hidden_size = config.get('hidden_size', 768)
        code_model = config.get('code_model', 'Salesforce/codet5p-770m')
        log_vocab_size = config.get('log_vocab_size', 10000)
        num_fault_types = config.get('num_fault_types', 9)  # K=8 + normal
        num_heads = config.get('num_heads', 8)

        # Encoders
        self.code_encoder = CodeChangeEncoder(model_name=code_model, hidden_size=hidden_size)
        self.context_encoder = FaultContextEncoder(vocab_size=log_vocab_size, hidden_size=hidden_size)

        # Fault Spectrum Alignment (Section 3.3)
        self.spectrum = FaultSpectrumAlignment(
            hidden_size=hidden_size, num_fault_types=num_fault_types
        )

        # Temporal Causal Fusion (Section 3.4) — replaces simple CrossModalFusion
        self.fusion = TemporalCausalFusion(
            hidden_size=hidden_size, num_heads=num_heads,
            num_fault_types=num_fault_types
        )

        # Contrastive Loss (Section 3.5)
        self.contrastive_loss = ContrastiveLoss(
            temperature=config.get('temperature', 0.07)
        )

        # Prediction head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )

        # Loss weights
        self.lambda_contrastive = config.get('lambda_contrastive', 0.1)
        self.lambda_align = config.get('lambda_align', 0.05)
        self.lambda_normal = config.get('lambda_normal', 0.02)

    def forward(self, code_input_ids: torch.Tensor, code_attention_mask: torch.Tensor,
                log_input_ids: torch.Tensor, log_attention_mask: torch.Tensor,
                labels: torch.Tensor = None, fault_types: torch.Tensor = None,
                timestamps: torch.Tensor = None) -> dict:
        """
        Args:
            code_input_ids: Tokenized code diff [batch, code_len]
            code_attention_mask: Code attention mask [batch, code_len]
            log_input_ids: Tokenized log templates [batch, log_len]
            log_attention_mask: Log attention mask [batch, log_len]
            labels: Binary fault labels [batch] (optional, for training)
            fault_types: Fault type indices [batch] (optional, for spectrum supervision)
            timestamps: Log timestamps [batch, log_len] (optional, for temporal decay)

        Returns:
            Dictionary with predictions, losses, and attention weights
        """
        # Encode code changes (returns pooled + token-level hidden states)
        code_emb, code_hidden, code_attn = self.code_encoder(code_input_ids, code_attention_mask)

        # Encode log context (returns pooled + token-level hidden states)
        context_emb, log_hidden, log_attn = self.context_encoder(
            log_input_ids, log_attention_mask, timestamps
        )

        # Fault Spectrum Alignment (uses pooled embeddings)
        P_c, P_r = self.spectrum(code_emb, context_emb)

        # Token-level Spectrum-Guided Cross-Modal Attention
        fused_emb = self.fusion(
            code_emb, context_emb, P_c, P_r,
            code_hidden=code_hidden, log_hidden=log_hidden,
            code_mask=code_attention_mask, log_mask=log_attention_mask
        )

        # Predict fault risk
        combined = torch.cat([code_emb, fused_emb], dim=-1)
        logits = self.classifier(combined).squeeze(-1)
        prob = torch.sigmoid(logits)

        result = {
            'logits': logits,
            'probability': prob,
            'code_attention': code_attn,
            'log_attention': log_attn,
            'code_embedding': code_emb,
            'fused_embedding': fused_emb,
            'P_c': P_c,
            'P_r': P_r,
        }

        # Compute losses during training
        if labels is not None:
            # L_BCE: binary cross-entropy
            loss_bce = F.binary_cross_entropy_with_logits(logits, labels.float())

            # L_contrastive: in-batch contrastive loss
            loss_contrastive = self.contrastive_loss(fused_emb, labels)

            # L_align: fault spectrum alignment loss
            loss_align = self.spectrum.alignment_loss(P_c, P_r, labels, fault_types)

            # L_normal: encourage benign samples toward normal class
            benign_mask = ~labels.bool()
            loss_normal = torch.tensor(0.0, device=logits.device)
            if benign_mask.any():
                normal_idx = self.spectrum.num_fault_types - 1
                target = torch.full((benign_mask.sum(),), normal_idx,
                                  device=logits.device, dtype=torch.long)
                loss_normal = F.cross_entropy(
                    self.spectrum.log_predictor(context_emb[benign_mask]),
                    target
                )

            # Combined loss
            total_loss = (loss_bce
                         + self.lambda_contrastive * loss_contrastive
                         + self.lambda_align * loss_align
                         + self.lambda_normal * loss_normal)

            result['loss'] = total_loss
            result['loss_bce'] = loss_bce
            result['loss_contrastive'] = loss_contrastive
            result['loss_align'] = loss_align
            result['loss_normal'] = loss_normal

        return result

    def get_explanation(self, code_attn: torch.Tensor, code_tokens: list,
                       P_c: torch.Tensor = None, top_k: int = 5) -> dict:
        """Extract dual-level explanation: fault type + top-k code tokens."""
        attn_scores = code_attn.cpu().numpy()
        top_indices = attn_scores.argsort()[-top_k:][::-1]
        token_explanations = [(code_tokens[i], float(attn_scores[i]))
                            for i in top_indices if i < len(code_tokens)]

        result = {'top_tokens': token_explanations}

        if P_c is not None:
            fault_type_names = [
                'NullReference', 'TypeMismatch', 'BoundaryError', 'ResourceLeak',
                'ConfigError', 'APIViolation', 'LogicError', 'Concurrency', 'Normal'
            ]
            probs = P_c.cpu().numpy()
            top_type_idx = probs.argmax()
            result['fault_type'] = fault_type_names[top_type_idx]
            result['fault_type_prob'] = float(probs[top_type_idx])

        return result


class ContrastiveLoss(nn.Module):
    """Contrastive loss for separating fault-inducing and benign changes."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [batch, hidden]
            labels: [batch] binary labels
            
        Returns:
            Contrastive loss
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        
        # Similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create masks
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.T).float()
        negative_mask = (labels != labels.T).float()
        
        # Remove diagonal
        mask = torch.eye(sim_matrix.size(0), device=sim_matrix.device)
        positive_mask = positive_mask * (1 - mask)
        negative_mask = negative_mask * (1 - mask)
        
        # Contrastive loss
        exp_sim = torch.exp(sim_matrix)
        
        pos_sum = (exp_sim * positive_mask).sum(dim=1)
        neg_sum = (exp_sim * negative_mask).sum(dim=1)
        
        loss = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8))
        
        return loss.mean()


if __name__ == "__main__":
    # Test model initialization
    print("Testing CodeFault model...")
    
    config = {
        'hidden_size': 768,
        'code_model': 'Salesforce/codet5p-220m',  # Smaller for testing
        'log_vocab_size': 10000
    }
    
    # Note: Full model requires downloading CodeT5+
    print("Model architecture defined successfully!")
    print(f"Config: {config}")
