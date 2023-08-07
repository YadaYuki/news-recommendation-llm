import torch
from transformers import AutoConfig
from src.recommendation.nrms.NRMS import NRMS
from src.recommendation.nrms.PLMBasedNewsEncoder import PLMBasedNewsEncoder
from src.recommendation.nrms.UserEncoder import UserEncoder


def test_nrms() -> None:
    pretrained: str = "bert-base-uncased"
    multihead_attn_num_heads: int = 16
    additive_attn_hidden_dim: int = 200
    batch_size, seq_len, hist_size, candidate_num, emb_dim = (
        20,
        10,
        4,
        5,
        AutoConfig.from_pretrained(pretrained).hidden_size,
    )

    plm_news_encoder = PLMBasedNewsEncoder(pretrained, multihead_attn_num_heads, additive_attn_hidden_dim)
    user_encoder = UserEncoder(emb_dim, multihead_attn_num_heads, additive_attn_hidden_dim)
    plm_based_nrms = NRMS(plm_news_encoder, user_encoder, emb_dim)

    candidate_news_batch = torch.arange(batch_size * candidate_num * seq_len).view(batch_size, candidate_num, seq_len)
    news_histories_batch = torch.arange(batch_size * hist_size * seq_len).view(batch_size, hist_size, seq_len)
    target = torch.randint(candidate_num, (batch_size,))

    plm_based_nrms.set_mode("train")
    model_output = plm_based_nrms(candidate_news_batch, news_histories_batch, target)
    assert tuple(model_output.logits.size()) == (batch_size, candidate_num)

    plm_based_nrms.set_mode("val")
    model_output = plm_based_nrms(candidate_news_batch, news_histories_batch, target)
    assert tuple(model_output.logits.size()) == (batch_size, candidate_num)
    assert float(model_output.loss) == -1.0
