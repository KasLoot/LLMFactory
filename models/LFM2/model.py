import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class LFM2_5_350M_Config:

    vocab_size = 64400 # LFM 2.5 VL tokenizer

    rope_base = 1000000.0

    d_model = 1024
    d_ff = 4608
    q_head = 16
    kv_head = 8
    d_head = 64

    conv_k = 3

    layers = [
        "conv",
        "conv",
        "gqa",
        "conv",
        "conv",
        "gqa",
        "conv",
        "conv",
        "gqa",
        "conv",
        "gqa",
        "conv",
        "gqa",
        "conv",
        "gqa",
        "conv"
        ]

    dr = 0.1
 







class RotaryEmbedding(nn.Module):
    """Rotary position embedding (RoPE).

    Precomputes the inverse frequencies and lazily caches the cos/sin tables for
    the longest sequence seen so far. The rotation is applied per head over the
    ``d_head`` feature dimension, which must be even.
    """

    def __init__(self, dim: int, base: float = 1000000.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension must be even"
        # inv_freq[i] = base^(-2i / dim) for i in [0, dim/2).
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: torch.Tensor = None
        self._sin_cached: torch.Tensor = None

    def _update_cache(self, seq_len: int, device, dtype):
        if (self._cos_cached is None or seq_len > self._seq_len_cached
                or self._cos_cached.device != device or self._cos_cached.dtype != dtype):
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)           # [seq_len, dim/2]
            emb = torch.cat([freqs, freqs], dim=-1)         # [seq_len, dim]
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)

    def forward(self, seq_len: int, device, dtype):
        self._update_cache(seq_len, device, dtype)
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the feature dim by swapping/negating its two halves: [-x2, x1]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply RoPE to query and key tensors.

    Args:
        q, k: ``[batch, head, seq_len, d_head]``.
        cos, sin: ``[seq_len, d_head]`` rotation tables.
    """
    cos = cos.unsqueeze(0).unsqueeze(0)                     # [1, 1, seq_len, d_head]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed




class GQA(nn.Module):
    def __init__(self, d_head, q_head, kv_head, dr, rope_base):
        super().__init__()
        self.d_head = d_head
        self.d_q = d_head * q_head
        self.d_kv = d_head * kv_head
        self.q_head = q_head
        self.kv_head = kv_head


        self.wq = nn.Linear(self.d_q, self.d_q)
        self.wk = nn.Linear(self.d_q, self.d_kv)
        self.wv = nn.Linear(self.d_q, self.d_kv)

        self.fc = nn.Linear(self.d_q, self.d_q, bias=False)

        self.dropout = nn.Dropout(dr)
        # RoPE rotates each head's d_head-dim q/k vectors by their token position.
        self.rotary = RotaryEmbedding(d_head, base=rope_base)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        # GQA: each kv head is shared by (q_head // kv_head) query heads.
        # Repeat kv heads so the head dim matches q before the matmul.
        n_rep = self.q_head // self.kv_head
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

        score = q @ k.transpose(-2, -1) / (self.d_head ** 0.5)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            score = score.masked_fill(~mask, torch.finfo(score.dtype).min)

        attn = F.softmax(score, dim=-1)
        attn = self.dropout(attn)
        output = attn @ v
        return output
    
    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor, mask: torch.Tensor = None):
        batch_size, seq_len, _ = x_q.size()

        q = self.wq(x_q).view(batch_size, seq_len, self.q_head, self.d_head).transpose(1, 2)
        k = self.wk(x_k).view(batch_size, seq_len, self.kv_head, self.d_head).transpose(1, 2)
        v = self.wv(x_v).view(batch_size, seq_len, self.kv_head, self.d_head).transpose(1, 2)

        # Inject positional information by rotating q/k before the attention scores.
        cos, sin = self.rotary(seq_len, x_q.device, q.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        output = self.attention(q, k, v, mask)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_q)
        output = self.fc(output)
        return output
    

class GSConv(nn.Module):
    def __init__(self, d_model, conv_k):
        super().__init__()
        self.conv_k = conv_k
        self.wbch = nn.Linear(d_model, 3*d_model)
        self.conv = nn.Conv1d(d_model, d_model, conv_k, bias=False, groups=d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        gate, channel, hidden = self.wbch(x).chunk(3, dim=-1)
        y = gate * hidden
        y = F.pad(y.transpose(-1, -2), (self.conv_k - 1, 0))
        z = self.conv(y).transpose(-1, -2).contiguous()
        out = self.out_proj(channel * z)

        return out



class SwiGLU(nn.Module):
 
    def __init__(self, d_model, d_ff, bias=False):
        super().__init__()
        # Replace the two separate projections with one fused projection.
        self.in_proj = nn.Linear(d_model, 2 * d_ff, bias=bias) # W, V
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)  # W2


    def forward(self, x):
        gate, up = self.in_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)



class TransformerBlock(nn.Module):
    def __init__(self, sequence_block, d_model, d_ff, d_head, q_head, kv_head, conv_k, dr, rope_base):
        super().__init__()
        self.sequence_block = sequence_block

        if sequence_block == "conv":
            self.sq_block = GSConv(d_model=d_model, conv_k=conv_k)
        elif sequence_block == "gqa":
            self.sq_block = GQA(d_head=d_head, q_head=q_head, kv_head=kv_head, dr=dr, rope_base=rope_base)
        else:
            raise ValueError(f"Unsupported sequence_block: {sequence_block}")
        
        self.sequence_norm = nn.RMSNorm(d_model)
        self.ffn_norm = nn.RMSNorm(d_model)

        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)

    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        sequence_input = self.sequence_norm(x)
        if self.sequence_block == "conv":
            x = x + self.sq_block(sequence_input)
        elif self.sequence_block == "gqa":
            x = x + self.sq_block(x_q=sequence_input, x_k=sequence_input, x_v=sequence_input, mask=mask)
        x = x + self.ffn(self.ffn_norm(x))

        return x






class LFM2(nn.Module):
    def __init__(self, config: LFM2_5_350M_Config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.q_head = config.q_head
        self.kv_head = config.kv_head
        self.d_head = config.d_head

        self.conv_k = config.conv_k

        self.layers = config.layers
        
        self.dr = config.dr
        self.rope_base = config.rope_base

        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.transformer = nn.ModuleList([
            TransformerBlock(sequence_block=sequence_block, d_model=self.d_model, d_ff=self.d_ff, d_head=self.d_head, q_head=self.q_head, kv_head=self.kv_head, conv_k=self.conv_k, dr=self.dr, rope_base=self.rope_base)
            for sequence_block in self.layers
        ])
        self.post_norm = nn.RMSNorm(self.d_model)
        self.out = nn.Linear(self.d_model, self.vocab_size, bias=False)

        self.apply(self._init_weights)
        self.out.weight = self.embedding.weight
        self.gradient_checkpointing = False

    
    # initialize transformer weights with a small std
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def set_gradient_checkpointing(self, enabled: bool = True):
        self.gradient_checkpointing = enabled


    def _make_causal_mask(self, seq_len: int, device: torch.device):
        return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()


    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        mask = self._make_causal_mask(x.size(1), x.device)
        for layer in self.transformer:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, mask, use_reentrant=False)
            else:
                x = layer(x, mask)
        x = self.out(self.post_norm(x))
        return x




if __name__ == "__main__":

    seq_len = 10
    batch_size = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():

        model_config = LFM2_5_350M_Config()
        x = torch.randint(low=0, high=model_config.vocab_size, size=(batch_size, seq_len)).to(device)
        model = LFM2(model_config).to(torch.bfloat16).to(device)
        model.eval()

        import torchinfo
        torchinfo.summary(model=model, input_data=x, device=device)

        print(f"Model Param Count: {sum(p.numel() for p in model.parameters())}")

        model_out = model(x)
        print(f"LFM2 model output shape: {model_out.shape}")
