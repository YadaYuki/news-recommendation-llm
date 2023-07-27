import torch
from transformers import AutoConfig
from src.recommendation.nrms.NRMS import NRMS
from src.recommendation.nrms.PLMBasedNewsEncoder import PLMBasedNewsEncoder
from src.recommendation.nrms.UserEncoder import UserEncoder


def test_nrms() -> None:
    pretrained: str = "bert-base-uncased"
    multihead_attn_num_heads: int = 16
    additive_attn_hidden_dim: int = 200
    batch_size, seq_len, hist_size, emb_dim = 20, 10, 4, AutoConfig.from_pretrained(pretrained).hidden_size
    plm_news_encoder = PLMBasedNewsEncoder(pretrained, multihead_attn_num_heads, additive_attn_hidden_dim)
    user_encoder = UserEncoder(emb_dim, multihead_attn_num_heads, additive_attn_hidden_dim)
    plm_based_nrms = NRMS(plm_news_encoder, user_encoder)

    candidate_news_batch = torch.arange(batch_size * seq_len).view(batch_size, seq_len)
    news_histories_batch = torch.arange(batch_size * hist_size * seq_len).view(batch_size, hist_size, seq_len)

    assert tuple(plm_based_nrms(candidate_news_batch, news_histories_batch).size()) == (batch_size,)
