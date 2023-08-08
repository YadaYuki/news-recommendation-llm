from transformers import AutoTokenizer
from mind.dataframe import read_behavior_df, read_news_df
from mind.MINDDataset import MINDValDataset
from const.path import MIND_SMALL_VAL_DATASET_DIR
from utils.random_seed import set_random_seed
from utils.text import create_transform_fn_from_pretrained_tokenizer
import torch
from utils.logger import logging
from evaluation.RecEvaluator import RecEvaluator, RecMetrics
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils.slack import notify_slack

set_random_seed()


def train(
    pretrained: str = "distilbert-base-uncased",
    npratio: int = 4,
    history_size: int = 50,
    batch_size: int = 32,
    epochs: int = 3,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    max_len: int = 30,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> None:
    logging.info("Start")
    """
    0. Definite Parameters & Functions
    """
    transform_fn = create_transform_fn_from_pretrained_tokenizer(AutoTokenizer.from_pretrained(pretrained), max_len)

    """
    1. Load Data & Create Dataset
    """
    val_news_df = read_news_df(MIND_SMALL_VAL_DATASET_DIR / "news.tsv")
    val_behavior_df = read_behavior_df(MIND_SMALL_VAL_DATASET_DIR / "behaviors.tsv")
    eval_dataset = MINDValDataset(val_behavior_df, val_news_df, transform_fn, history_size)

    eval_dataloader = DataLoader(eval_dataset, batch_size=1, pin_memory=True)
    metrics_average = RecMetrics(
        **{
            "ndcg_at_10": 0.0,
            "ndcg_at_5": 0.0,
            "auc": 0.0,
            "mrr": 0.0,
        }
    )
    for batch in tqdm(eval_dataloader, desc="Evaluation for MINDValDataset"):
        y_true: np.ndarray = batch["target"].flatten().cpu().to(torch.int).numpy()
        y_score: np.ndarray = np.random.rand(len(y_true))

        metrics = RecEvaluator.evaluate_all(y_true, y_score)
        metrics_average.ndcg_at_10 += metrics.ndcg_at_10
        metrics_average.ndcg_at_5 += metrics.ndcg_at_5
        metrics_average.auc += metrics.auc
        metrics_average.mrr += metrics.mrr

    metrics_average.ndcg_at_10 /= len(eval_dataset)
    metrics_average.ndcg_at_5 /= len(eval_dataset)
    metrics_average.auc /= len(eval_dataset)
    metrics_average.mrr /= len(eval_dataset)

    logging.info(metrics_average.dict())

    notify_slack(
        f"""```
    {metrics_average.dict()}
    ```
    """
    )


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        notify_slack(
            f"""```
                {e}
            ```
            """
        )
