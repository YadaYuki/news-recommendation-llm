from transformers import TrainingArguments, AutoConfig, Trainer, AutoTokenizer
from recommendation.nrms import PLMBasedNewsEncoder, NRMS, UserEncoder
from torch import nn
from mind.dataframe import read_behavior_df, read_news_df
from mind.MINDDataset import MINDTrainDataset, MINDValDataset
from const.path import MIND_SMALL_TRAIN_DATASET_DIR, MIND_SMALL_VAL_DATASET_DIR, MODEL_OUTPUT_DIR, LOG_OUTPUT_DIR
from utils.random_seed import set_random_seed
from utils.text import create_transform_fn_from_pretrained_tokenizer
import torch
from utils.logger import logging
from evaluation.RecEvaluator import RecEvaluator, RecMetrics
from torch.utils.data import DataLoader
from transformers.modeling_outputs import ModelOutput
from tqdm import tqdm
from utils.slack import notify_slack

set_random_seed()


def train(
    pretrained: str = "distilbert-base-uncased",
    npratio: int = 4,
    history_size: int = 20,
    batch_size: int = 32,
    epochs: int = 1,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.0,
    max_len: int = 16,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> None:
    logging.info("Start")
    """
    0. Definite Parameters & Functions
    """
    hidden_size: int = AutoConfig.from_pretrained(pretrained).hidden_size
    loss_fn: nn.Module = nn.CrossEntropyLoss()
    transform_fn = create_transform_fn_from_pretrained_tokenizer(AutoTokenizer.from_pretrained(pretrained), max_len)
    EVAL_BATCH_SIZE = 1

    """
    1. Init Model
    """
    logging.info("Initilize Model")
    news_encoder = PLMBasedNewsEncoder(pretrained)
    user_encoder = UserEncoder(hidden_size=hidden_size)
    nrms_net = NRMS(news_encoder=news_encoder, user_encoder=user_encoder, hidden_size=hidden_size, loss_fn=loss_fn).to(
        device, dtype=torch.bfloat16
    )

    """
    2. Load Data & Create Dataset
    """
    logging.info("Initilize Dataset")
    train_news_df = read_news_df(MIND_SMALL_TRAIN_DATASET_DIR / "news.tsv")
    train_behavior_df = read_behavior_df(MIND_SMALL_TRAIN_DATASET_DIR / "behaviors.tsv")
    val_news_df = read_news_df(MIND_SMALL_VAL_DATASET_DIR / "news.tsv")
    val_behavior_df = read_behavior_df(MIND_SMALL_VAL_DATASET_DIR / "behaviors.tsv")
    train_dataset = MINDTrainDataset(train_behavior_df, train_news_df, transform_fn, npratio, history_size, device)
    eval_dataset = MINDValDataset(val_behavior_df, val_news_df, transform_fn, history_size)

    """
    3. Train
    """
    logging.info("Training Start")
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        logging_strategy="steps",
        save_total_limit=5,
        lr_scheduler_type="constant",
        weight_decay=weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=epochs,
        remove_unused_columns=False,
        logging_dir=LOG_OUTPUT_DIR,
        logging_steps=5,
        report_to="none",
    )

    trainer = Trainer(
        model=nrms_net,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()

    """
    4. Evaluate
    """
    trainer.model.eval()
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
        batch["news_histories"] = batch["news_histories"].to(device)
        batch["candidate_news"] = batch["candidate_news"].to(device)
        batch["target"] = batch["target"].to(device)

        with torch.no_grad():
            model_output: ModelOutput = trainer.model(**batch)

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
