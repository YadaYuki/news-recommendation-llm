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
import hydra
from utils.slack import notify_slack
from config.config import TrainConfig
from utils.path import generate_folder_name_with_timestamp


def train(
    pretrained: str,
    npratio: int,
    history_size: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    max_len: int,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> None:
    logging.info("Start")
    """
    0. Definite Parameters & Functions
    """
    EVAL_BATCH_SIZE = 1
    hidden_size: int = AutoConfig.from_pretrained(pretrained).hidden_size
    loss_fn: nn.Module = nn.CrossEntropyLoss()
    transform_fn = create_transform_fn_from_pretrained_tokenizer(AutoTokenizer.from_pretrained(pretrained), max_len)
    model_save_dir = generate_folder_name_with_timestamp(MODEL_OUTPUT_DIR)

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
    train_dataset = MINDTrainDataset(train_behavior_df, train_news_df, transform_fn, npratio, history_size, device)

    val_news_df = read_news_df(MIND_SMALL_VAL_DATASET_DIR / "news.tsv")
    val_behavior_df = read_behavior_df(MIND_SMALL_VAL_DATASET_DIR / "behaviors.tsv")
    eval_dataset = MINDValDataset(val_behavior_df, val_news_df, transform_fn, history_size)

    """
    3. Train
    """
    logging.info("Training Start")
    training_args = TrainingArguments(
        output_dir=model_save_dir,
        logging_strategy="steps",
        save_total_limit=5,
        lr_scheduler_type="constant",
        weight_decay=weight_decay,
        optim="adamw_torch",
        evaluation_strategy="no",
        save_strategy="epoch",
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=epochs,
        remove_unused_columns=False,
        logging_dir=LOG_OUTPUT_DIR,
        logging_steps=1,
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
    eval_dataloader = DataLoader(eval_dataset, batch_size=EVAL_BATCH_SIZE, pin_memory=True)
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
        pretrained:{pretrained}
        npratio:{npratio}
        history_size:{history_size}
        batch_size:{batch_size}
        gradient_accumulation_steps:{gradient_accumulation_steps}
        epochs:{epochs}
        learning_rate:{learning_rate}
        weight_decay:{weight_decay}
        max_len:{max_len}
        device:{device}
        {metrics_average.dict()}
    ```

    """
    )


@hydra.main(version_base=None, config_name="train_config")
def main(cfg: TrainConfig) -> None:
    try:
        set_random_seed(cfg.random_seed)
        train(
            cfg.pretrained,
            cfg.npratio,
            cfg.history_size,
            cfg.batch_size,
            cfg.gradient_accumulation_steps,
            cfg.epochs,
            cfg.learning_rate,
            cfg.weight_decay,
            cfg.max_len,
        )
    except Exception as e:
        notify_slack(f"```{e}```")


if __name__ == "__main__":
    main()
