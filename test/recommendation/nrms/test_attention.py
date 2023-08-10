import torch

from src.recommendation.nrms.AdditiveAttention import AdditiveAttention


def test_additive_attention() -> None:
    batch_size, seq_len, emb_dim, hidden_dim = 20, 10, 30, 5
    attn = AdditiveAttention(emb_dim, hidden_dim)
    input = torch.rand(batch_size, seq_len, emb_dim)
    assert tuple(attn(input).shape) == (batch_size, seq_len, emb_dim)
