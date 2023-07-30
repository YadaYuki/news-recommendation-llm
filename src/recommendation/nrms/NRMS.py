from torch import nn
import torch
from transformers.modeling_outputs import ModelOutput


class NRMS(nn.Module):
    def __init__(
        self,
        news_encoder: nn.Module,
        user_encoder: nn.Module,
        hidden_size: int,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
    ) -> None:
        super().__init__()
        self.news_encoder: nn.Module = news_encoder
        self.user_encoder: nn.Module = user_encoder
        self.hidden_size: int = hidden_size
        self.loss_fn = loss_fn

        self.__mode: str | None = None

    def forward(
        self, candidate_news: torch.Tensor, news_histories: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        candidate_news : torch.Tensor (shape = (batch_size, candidate_num, seq_len))
        news_histories : torch.Tensor (shape = (batch_size, candidate_num, seq_len))
        ===========================================================================

        Returns
        ----------
        output: torch.Tensor (shape = (batch_size, candidate_num))

        """
        assert self.__mode in ["val", "train"]

        batch_size, candidate_num, seq_len = candidate_news.size()
        candidate_news = candidate_news.view(batch_size * candidate_num, seq_len)
        news_candidate_encoded = self.news_encoder(
            candidate_news
        )  # [batch_size * (candidate_num), seq_len] -> [batch_size * (candidate_num), emb_dim]
        news_candidate_encoded = news_candidate_encoded.view(
            batch_size, candidate_num, self.hidden_size
        )  # [batch_size * (candidate_num), emb_dim] -> [batch_size, (candidate_num), emb_dim]

        news_histories_encoded = self.user_encoder(
            news_histories, self.news_encoder
        )  # [batch_size, histories, seq_len] -> [batch_size, emb_dim]
        news_histories_encoded = news_histories_encoded.unsqueeze(
            -1
        )  # [batch_size, emb_dim] -> [batch_size, emb_dim, 1]

        output = torch.bmm(
            news_candidate_encoded, news_histories_encoded
        )  # [batch_size, (candidate_num), emb_dim] x [batch_size, emb_dim, 1] -> [batch_size, (1+npratio), 1, 1]
        output = output.squeeze(-1).squeeze(-1)  # [batch_size, (1+npratio), 1, 1] -> [batch_size, (1+npratio)]

        # NOTE:
        # when "val" mode → not calculate loss score
        # Multiple hot labels may exist on target.
        # e.g.
        # candidate_news = ["N24510","N39237","N9721"]
        # target = [0,2]( = [1, 0, 1] in one-hot format)
        if self.__mode == "val":
            return ModelOutput(logits=output, loss=-1)

        loss = self.loss_fn(output, target)
        return ModelOutput(logits=output, loss=loss)

    def set_mode(self, mode: str) -> None:
        assert mode in ["val", "train"]
        self.__mode = mode
