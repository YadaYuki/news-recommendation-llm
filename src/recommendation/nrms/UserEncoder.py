from torch import nn
from .AdditiveAttention import AdditiveAttention
import torch


class UserEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        multihead_attn_num_heads: int = 16,
        additive_attn_hidden_dim: int = 200,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=multihead_attn_num_heads, batch_first=True
        )
        self.additive_attention = AdditiveAttention(hidden_size, additive_attn_hidden_dim)

    def forward(self, news_histories: torch.Tensor, news_encoder: nn.Module) -> torch.Tensor:
        batch_size, hist_size, seq_len = news_histories.size()
        news_histories = news_histories.view(
            batch_size * hist_size, seq_len
        )  # [batch_size, hist_size, seq_len] -> [batch_size*hist_size, seq_len]

        news_histories_encoded = news_encoder(
            news_histories
        )  # [batch_size*hist_size, seq_len] -> [batch_size*hist_size, emb_dim]

        news_histories_encoded = news_histories_encoded.view(
            batch_size, hist_size, self.hidden_size
        )  # [batch_size*hist_size, seq_len] -> [batch_size, hist_size, emb_dim]

        multihead_attn_output, _ = self.multihead_attention(
            news_histories_encoded, news_histories_encoded, news_histories_encoded
        )  # [batch_size, hist_size, emb_dim] -> [batch_size, hist_size, emb_dim]

        additive_attn_output = self.additive_attention(
            multihead_attn_output
        )  # [batch_size, hist_size, emb_dim] -> [batch_size, emb_dim]

        output = torch.sum(additive_attn_output, dim=1)

        return output
