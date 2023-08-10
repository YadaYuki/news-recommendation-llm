import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from .AdditiveAttention import AdditiveAttention


class PLMBasedNewsEncoder(nn.Module):
    def __init__(
        self,
        pretrained: str = "bert-base-uncased",
        multihead_attn_num_heads: int = 16,
        additive_attn_hidden_dim: int = 200,
    ):
        super().__init__()
        self.plm = AutoModel.from_pretrained(pretrained)

        plm_hidden_size = AutoConfig.from_pretrained(pretrained).hidden_size

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=plm_hidden_size, num_heads=multihead_attn_num_heads, batch_first=True
        )
        self.additive_attention = AdditiveAttention(plm_hidden_size, additive_attn_hidden_dim)

    def forward(self, input_val: torch.Tensor) -> torch.Tensor:
        V = self.plm(input_val).last_hidden_state  # [batch_size, seq_len] -> [batch_size, seq_len, hidden_size]
        multihead_attn_output, _ = self.multihead_attention(
            V, V, V
        )  # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, hidden_size]
        additive_attn_output = self.additive_attention(
            multihead_attn_output
        )  # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, hidden_size]
        output = torch.sum(
            additive_attn_output, dim=1
        )  # [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size]

        return output
