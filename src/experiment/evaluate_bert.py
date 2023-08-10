import torch
from const.path import MIND_SMALL_VAL_DATASET_DIR, MODEL_OUTPUT_DIR
from evaluation.RecEvaluator import RecEvaluator, RecMetrics
from mind.dataframe import read_behavior_df, read_news_df
from mind.MINDDataset import MINDValDataset
from recommendation.nrms import NRMS, PLMBasedNewsEncoder, UserEncoder
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from transformers.modeling_outputs import ModelOutput
from utils.logger import logging
from utils.random_seed import set_random_seed
from utils.slack import notify_slack
from utils.text import create_transform_fn_from_pretrained_tokenizer


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
    hidden_size: int = AutoConfig.from_pretrained(pretrained).hidden_size
    loss_fn: nn.Module = nn.CrossEntropyLoss()

    """
    1. Load Data & Create Dataset
    """
    val_news_df = read_news_df(MIND_SMALL_VAL_DATASET_DIR / "news.tsv")
    val_behavior_df = read_behavior_df(MIND_SMALL_VAL_DATASET_DIR / "behaviors.tsv")
    eval_dataset = MINDValDataset(val_behavior_df, val_news_df, transform_fn, history_size)

    eval_dataloader = DataLoader(eval_dataset, batch_size=1, pin_memory=True)

    """
    2. Load Model
    """

    logging.info("Initilize Model")
    news_encoder = PLMBasedNewsEncoder(pretrained)
    user_encoder = UserEncoder(hidden_size=hidden_size)
    nrms_net = NRMS(news_encoder=news_encoder, user_encoder=user_encoder, hidden_size=hidden_size, loss_fn=loss_fn).to(
        device, dtype=torch.bfloat16
    )
    path_to_model = MODEL_OUTPUT_DIR / "./checkpoint-3678/training_args.bin"
    nrms_net.load_state_dict(torch.load(path_to_model))

    """
    3. Evaluate
    """

    metrics_average = RecMetrics(
        **{
            "ndcg_at_10": 0.0,
            "ndcg_at_5": 0.0,
            "auc": 0.0,
            "mrr": 0.0,
        }
    )
    for batch in tqdm(eval_dataloader, desc="Evaluation for MINDValDataset"):
        batch["news_histories"] = batch["news_histories"].to(device)
        batch["candidate_news"] = batch["candidate_news"].to(device)
        batch["target"] = batch["target"].to(device)

        with torch.no_grad():
            model_output: ModelOutput = nrms_net(**batch)

        y_score: torch.Tensor = model_output.logits.flatten().cpu().to(torch.float64).numpy()
        y_true: torch.Tensor = batch["target"].flatten().cpu().to(torch.int).numpy()

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
