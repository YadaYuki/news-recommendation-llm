from torch import nn
import torch


class NRMS(nn.Module):
    def __init__(self, news_encoder: nn.Module, user_encoder: nn.Module) -> None:
        super().__init__()
        self.news_encoder: nn.Module = news_encoder
        self.user_encoder: nn.Module = user_encoder

    def forward(self, candidate_news: torch.Tensor, news_histories: torch.Tensor) -> torch.Tensor:
        news_encoded = self.news_encoder(candidate_news)  # [batch_size, seq_len] -> [batch_size, emb_dim]
        news_histories_encoded = self.user_encoder(
            news_histories, self.news_encoder
        )  # [batch_size, histories, seq_len] -> [batch_size, emb_dim]

        return torch.sum(news_encoded * news_histories_encoded, dim=1)
