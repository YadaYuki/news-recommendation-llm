from torch import nn
import torch


class NRMS(nn.Module):
    def __init__(
        self,
        news_encoder: nn.Module,
        user_encoder: nn.Module,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.news_encoder: nn.Module = news_encoder
        self.user_encoder: nn.Module = user_encoder
        self.hidden_size: int = hidden_size

    def forward(self, candidate_news: torch.Tensor, news_histories: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        candidate_news : torch.Tensor (shape = (batch_size, 1 + npratio, seq_len))
        news_histories : torch.Tensor (shape = (batch_size, 1 + npratio, seq_len))
        Returns
        ----------
        output: torch.Tensor (shape = (batch_size, 1 + npratio))

        """

        batch_size, candidate_num, seq_len = candidate_news.size()
        candidate_news = candidate_news.view(batch_size * candidate_num, seq_len)
        news_candidate_encoded = self.news_encoder(
            candidate_news
        )  # [batch_size * (1 + npratio), seq_len] -> [batch_size * (1 + npratio), emb_dim]
        news_candidate_encoded = news_candidate_encoded.view(
            batch_size, candidate_num, self.hidden_size
        )  # [batch_size * (1 + npratio), emb_dim] -> [batch_size, (1 + npratio), emb_dim]

        news_histories_encoded = self.user_encoder(
            news_histories, self.news_encoder
        )  # [batch_size, histories, seq_len] -> [batch_size, emb_dim]
        news_histories_encoded = news_histories_encoded.unsqueeze(
            -1
        )  # [batch_size, emb_dim] -> [batch_size, emb_dim, 1]

        output = torch.bmm(
            news_candidate_encoded, news_histories_encoded
        )  # [batch_size, (1 + npratio), emb_dim] x [batch_size, emb_dim, 1] -> [batch_size, (1+npratio), 1, 1]

        return output.squeeze(-1).squeeze(-1)
