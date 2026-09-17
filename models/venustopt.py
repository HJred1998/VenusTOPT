"""VenusTOPT prediction head for PRIME residue representations."""
from __future__ import annotations

import json
import math
from pathlib import Path
import torch
from torch import nn
from .motif import LocalGlobalPooling, masked_stats_pool


class VenusTOPT(nn.Module):
    """Map masked residue embeddings to standardized Topt; no PLM inside this head."""

    def __init__(self, config: dict):
        super().__init__()
        dim = config['hidden_dim']
        self.input_norm = nn.LayerNorm(config['input_dim'])
        self.input_proj = nn.Linear(config['input_dim'], dim)
        scales = config['motif_scales']
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed((torch.initial_seed() + 1001) % (2**63 - 1))
            self.pooling = LocalGlobalPooling(
                dim, config['num_queries'], tuple(map(tuple, scales)),
                config['num_heads'], config['ffn_multiplier'], config['dropout'],
                config['cross_attention_layers'],
            )
        width = self.pooling.output_dim + 3 * dim
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed((torch.initial_seed() + 2001) % (2**63 - 1))
            layers = [nn.LayerNorm(width)]
            for _ in range(config['head_layers']):
                layers.extend([nn.Linear(width, config['head_hidden_dim']), nn.GELU(), nn.Dropout(config['dropout'])])
                width = config['head_hidden_dim']
            layers.append(nn.Linear(width, 1))
            self.head = nn.Sequential(*layers)

    def forward(self, x, mask, return_aux=False):
        if x.ndim != 3 or mask.shape != x.shape[:2] or mask.dtype != torch.bool:
            raise ValueError('Expected x [B,L,D] and boolean mask [B,L]')
        if not mask.any(dim=1).all():
            raise ValueError('Each sequence must contain at least one valid residue')
        h = self.input_proj(self.input_norm(x)).masked_fill(~mask.unsqueeze(-1), 0.0)
        result = self.pooling(h, mask, return_aux=return_aux)
        features, aux = result if return_aux else (result, {})
        stats = masked_stats_pool(h, mask)
        prediction = self.head(torch.cat([features, stats], dim=-1)).squeeze(-1)
        return (prediction, {**aux, 'global_stats': stats}) if return_aux else prediction


def load_predictor(checkpoint, config_path, device='cpu'):
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    if not math.isfinite(config['target_mean']) or not math.isfinite(config['target_std']) or config['target_std'] <= 0:
        raise ValueError('Model configuration requires finite target normalization with positive standard deviation')
    model = VenusTOPT(config)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model, config
