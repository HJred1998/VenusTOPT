"""Local PRIME encoder; residue representations stay in memory by default."""
from contextlib import nullcontext
import torch


class PrimeEncoder:
    def __init__(self, model_dir, device='cpu', precision='fp32'):
        from transformers import AutoModel, AutoTokenizer
        self.device = torch.device(device)
        self.precision = precision
        self.model = AutoModel.from_pretrained(
            str(model_dir), trust_remote_code=True, local_files_only=True,
        ).eval().to(self.device)
        if hasattr(self.model, 'contact_head'):
            self.model.contact_head = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=True, local_files_only=True,
        )
        # Reserve special tokens and keep the supported sequence interface explicit.
        self.max_residues = min(1022, self.model.config.max_position_embeddings - 2)

    @torch.inference_mode()
    def encode(self, sequence):
        if len(sequence) > self.max_residues:
            raise ValueError(f'Sequence has {len(sequence)} residues; supported maximum is {self.max_residues}. No automatic truncation.')
        encoded = self.tokenizer(sequence, return_tensors='pt', truncation=False)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        if encoded['input_ids'].shape[1] != len(sequence) + 2:
            raise ValueError('Tokenizer must produce one token per residue plus BOS/EOS')
        context = nullcontext()
        if self.device.type == 'cuda' and self.precision != 'fp32':
            dtype = torch.float16 if self.precision == 'fp16' else torch.bfloat16
            context = torch.autocast('cuda', dtype=dtype)
        with context:
            hidden = self.model.pro_prime(**encoded).last_hidden_state
        x = hidden[:, 1:-1].float()
        if not torch.isfinite(x).all():
            raise ValueError('Nonfinite PRIME embeddings; retry with --precision fp32')
        return x
