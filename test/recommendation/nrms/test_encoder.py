import torch
from transformers import AutoConfig

from src.recommendation.nrms.PLMBasedNewsEncoder import PLMBasedNewsEncoder
from src.recommendation.nrms.UserEncoder import UserEncoder


def test_news_encoder() -> None:
    pretrained: str = "bert-base-uncased"
    last_attn_num_heads: int = 12
    additive_attn_hidden_dim: int = 200
    plm_news_encoder = PLMBasedNewsEncoder(pretrained, last_attn_num_heads, additive_attn_hidden_dim)

    batch_size, seq_len, emb_dim = 20, 10, AutoConfig.from_pretrained(pretrained).hidden_size

    input_tensor = torch.arange(batch_size * seq_len).view(batch_size, seq_len)

    output = plm_news_encoder(input_tensor)

    assert tuple(output.size()) == (batch_size, emb_dim)


def test_user_encoder() -> None:
    pretrained: str = "bert-base-uncased"
    multihead_attn_num_heads: int = 12
    additive_attn_hidden_dim: int = 200
    hist_size: int = 4

    plm_news_encoder = PLMBasedNewsEncoder(pretrained, multihead_attn_num_heads, additive_attn_hidden_dim)

    batch_size, seq_len, emb_dim = 20, 10, AutoConfig.from_pretrained(pretrained).hidden_size

    news_histories = torch.arange(batch_size * hist_size * seq_len).view(batch_size, hist_size, seq_len)

    user_encoder = UserEncoder(emb_dim, multihead_attn_num_heads, additive_attn_hidden_dim)

    output = user_encoder(news_histories, plm_news_encoder)

    assert tuple(output.size()) == (batch_size, emb_dim)
