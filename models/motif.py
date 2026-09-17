"""Multi-scale motif tokenization and local-global attention pooling."""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F

class MotifCrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention from protein queries to local motif tokens."""

    def __init__(self, dim: int, num_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.motif_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        motifs: torch.Tensor,
        motif_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        normalized_motifs = self.motif_norm(motifs)
        update, attention = self.cross_attention(
            self.query_norm(queries),
            normalized_motifs,
            normalized_motifs,
            key_padding_mask=~motif_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        update = self.cross_dropout(update)
        update = update + self.ffn(self.ffn_norm(update))
        queries = queries + update
        return queries, update, attention if return_attention else None

class LocalGlobalPooling(nn.Module):
    """Multi-scale local motif tokens read by independent global protein queries."""

    def __init__(
        self,
        dim: int,
        num_queries: int,
        scales: tuple[tuple[int, int], ...],
        num_heads: int,
        ff_mult: int,
        dropout: float,
        cross_layers: int,
    ):
        super().__init__()
        if not scales:
            raise ValueError("At least one motif scale is required")
        if any(window < 2 or stride < 1 for window, stride in scales):
            raise ValueError(f"Invalid window/stride pairs: {scales}")
        if cross_layers < 1:
            raise ValueError("At least one cross-attention layer is required")
        self.dim = dim
        self.num_queries = num_queries
        self.scales = scales

        self.global_queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.global_score_norm = nn.LayerNorm(dim)
        self.global_dropout = nn.Dropout(dropout)

        self.motif_score_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim // 2),
                    nn.GELU(),
                    nn.Linear(dim // 2, 1),
                )
                for _ in scales
            ]
        )
        self.motif_value_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim),
                    nn.GELU(),
                )
                for _ in scales
            ]
        )
        self.scale_embeddings = nn.Parameter(torch.randn(len(scales), dim) * 0.02)
        self.position_projection = nn.Sequential(
            nn.Linear(2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.motif_output_norm = nn.LayerNorm(dim)

        self.cross_blocks = nn.ModuleList(
            [
                MotifCrossAttentionBlock(dim, num_heads, ff_mult, dropout)
                for _ in range(cross_layers)
            ]
        )
        self.local_feature_norm = nn.LayerNorm(dim)

    @property
    def output_dim(self) -> int:
        return 2 * self.num_queries * self.dim

    @staticmethod
    def _pad_for_windows(
        x: torch.Tensor,
        mask: torch.Tensor,
        window: int,
        stride: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        length = x.size(1)
        target = max(length, window)
        target += (stride - (target - window) % stride) % stride
        pad = target - length
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
            mask = F.pad(mask, (0, pad), value=False)
        return x, mask

    def _motif_tokens(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        return_attention: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        token_parts: list[torch.Tensor] = []
        mask_parts: list[torch.Tensor] = []
        attention_parts: list[torch.Tensor] = []
        start_parts: list[torch.Tensor] = []
        lengths = mask.sum(dim=1).to(x.dtype).clamp_min(1.0)

        for scale_index, ((window, stride), scorer, value_projection) in enumerate(
            zip(self.scales, self.motif_score_heads, self.motif_value_projections)
        ):
            score_sequence = scorer(x).squeeze(-1)
            value_sequence = value_projection(x)
            padded_values, padded_mask = self._pad_for_windows(
                value_sequence, mask, window, stride
            )
            padded_scores, _ = self._pad_for_windows(
                score_sequence.unsqueeze(-1), mask, window, stride
            )
            value_windows = padded_values.unfold(1, window, stride).permute(0, 1, 3, 2)
            score_windows = padded_scores.squeeze(-1).unfold(1, window, stride)
            window_mask = padded_mask.unfold(1, window, stride)
            valid = window_mask.any(dim=-1)

            logits = score_windows.masked_fill(
                ~window_mask,
                torch.finfo(score_windows.dtype).min,
            )
            weights = torch.softmax(logits, dim=-1)
            weights = weights * window_mask.to(weights.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            tokens = torch.sum(weights.unsqueeze(-1) * value_windows, dim=2)

            starts = torch.arange(
                tokens.size(1), device=x.device, dtype=x.dtype
            ) * float(stride)
            centers = starts + 0.5 * float(window - 1)
            center_fraction = centers.unsqueeze(0) / (lengths.unsqueeze(1) - 1.0).clamp_min(1.0)
            center_fraction = center_fraction.clamp(0.0, 1.0)
            scale_fraction = (
                math.log1p(float(window))
                / torch.log1p(lengths).clamp_min(math.log(2.0))
            ).unsqueeze(1).expand_as(center_fraction)
            position = self.position_projection(
                torch.stack([center_fraction, scale_fraction], dim=-1)
            )
            tokens = tokens + position + self.scale_embeddings[scale_index]
            tokens = tokens.masked_fill(~valid.unsqueeze(-1), 0.0)

            token_parts.append(tokens)
            mask_parts.append(valid)
            if return_attention:
                attention_parts.append(weights)
                start_parts.append(starts.to(torch.long))

        motifs = self.motif_output_norm(torch.cat(token_parts, dim=1))
        motif_mask = torch.cat(mask_parts, dim=1)
        motifs = motifs.masked_fill(~motif_mask.unsqueeze(-1), 0.0)
        return motifs, motif_mask, attention_parts, start_parts

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
        normalized = self.global_score_norm(x)
        global_logits = torch.einsum(
            "qd,bld->bql", self.global_queries, normalized
        ) / math.sqrt(self.dim)
        global_logits = global_logits.masked_fill(
            ~mask.unsqueeze(1), torch.finfo(global_logits.dtype).min
        )
        global_attention = torch.softmax(global_logits, dim=-1)
        global_tokens = torch.einsum(
            "bql,bld->bqd", self.global_dropout(global_attention), x
        )
        motif_aware = global_tokens

        motifs, motif_mask, motif_attention, motif_starts = self._motif_tokens(
            x, mask, return_aux
        )
        local_updates: list[torch.Tensor] = []
        cross_attention = None
        for block_index, block in enumerate(self.cross_blocks):
            motif_aware, local_update, block_attention = block(
                motif_aware,
                motifs,
                motif_mask,
                return_attention=return_aux and block_index == len(self.cross_blocks) - 1,
            )
            local_updates.append(local_update)
            if block_attention is not None:
                cross_attention = block_attention
        local_delta = torch.stack(local_updates).sum(dim=0)
        local_features = self.local_feature_norm(local_delta)
        output = torch.cat(
            [global_tokens.flatten(1), local_features.flatten(1)], dim=-1
        )

        if not return_aux:
            return output
        return output, {
            "attention": global_attention,
            "global_attention": global_attention,
            "global_tokens": global_tokens,
            "motif_tokens": motifs,
            "motif_mask": motif_mask,
            "motif_window_attention": motif_attention,
            "motif_window_starts": motif_starts,
            "motif_scales": self.scales,
            "cross_attention": cross_attention,
            "local_delta": local_delta,
            "local_features": local_features,
        }

def masked_stats_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).to(x.dtype)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    mean = (x * mask_f).sum(dim=1) / denom
    x_for_max = x.masked_fill(~mask.unsqueeze(-1), torch.finfo(x.dtype).min)
    max_pool = x_for_max.max(dim=1).values
    var = (((x - mean.unsqueeze(1)) ** 2) * mask_f).sum(dim=1) / denom
    std = torch.sqrt(var.clamp_min(1e-8))
    return torch.cat([mean, max_pool, std], dim=-1)
