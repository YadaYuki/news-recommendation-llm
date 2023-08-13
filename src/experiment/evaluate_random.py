import numpy as np
import torch
from const.path import MIND_SMALL_VAL_DATASET_DIR
from evaluation.RecEvaluator import RecEvaluator, RecMetrics
from mind.dataframe import read_behavior_df, read_news_df
from mind.MINDDataset import MINDValDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.logger import logging
from utils.random_seed import set_random_seed
from utils.text import create_transform_fn_from_pretrained_tokenizer


set_random_seed()


def evaluate_random() -> None:
    logging.info("Start")

    """
    0. Definite Parameters & Functions
    """
    DUMMIY_VALUE = 1
    transform_fn = create_transform_fn_from_pretrained_tokenizer(
        AutoTokenizer.from_pretrained("distilbert-base-uncased"), DUMMIY_VALUE
    )

    """
    1. Load Data & Create Dataset
    """
    val_news_df = read_news_df(MIND_SMALL_VAL_DATASET_DIR / "news.tsv")
    val_behavior_df = read_behavior_df(MIND_SMALL_VAL_DATASET_DIR / "behaviors.tsv")
    eval_dataset = MINDValDataset(val_behavior_df, val_news_df, transform_fn, DUMMIY_VALUE)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, pin_memory=True)

    """
    2. Evaluation
    """
    val_metrics_list: list[RecMetrics] = []
    for batch in tqdm(eval_dataloader, desc="Evaluation for MINDValDataset"):
        y_true: np.ndarray = batch["target"].flatten().cpu().to(torch.int).numpy()
        y_score: np.ndarray = np.random.rand(len(y_true))  # Calculate score by rand
        # Calculate Metrics
        val_metrics_list.append(RecEvaluator.evaluate_all(y_true, y_score))

    rec_metrics = RecMetrics(
        **{
            "ndcg_at_10": np.average([metrics_item.ndcg_at_10 for metrics_item in val_metrics_list]),
            "ndcg_at_5": np.average([metrics_item.ndcg_at_5 for metrics_item in val_metrics_list]),
            "auc": np.average([metrics_item.auc for metrics_item in val_metrics_list]),
            "mrr": np.average([metrics_item.mrr for metrics_item in val_metrics_list]),
        }
    )

    logging.info(rec_metrics.dict())


if __name__ == "__main__":
    try:
        evaluate_random()
    except Exception as e:
        logging.error(e)
