from transformers import TrainingArguments, AutoConfig, Trainer, AutoTokenizer, EvalPrediction
from recommendation.nrms import PLMBasedNewsEncoder, NRMS, UserEncoder
from torch import nn
from mind.dataframe import read_behavior_df, read_news_df
from mind.MINDDataset import MINDTrainDataset, MINDValDataset
from const.path import MIND_SMALL_TRAIN_DATASET_DIR, MIND_SMALL_VAL_DATASET_DIR, MODEL_OUTPUT_DIR
from utils.random_seed import set_random_seed
from utils.text import create_transform_fn_from_pretrained_tokenizer
import torch
from utils.logger import logging
from evaluation.RecEvaluator import RecEvaluator

set_random_seed()


def compute_rec_metrics(pred: EvalPrediction) -> dict[str, float]:
    y_score, y_true = pred.predictions, pred.label_ids
    assert y_score.shape == y_true.shape

    batch_size, candidate_num = y_score.shape
    assert batch_size == 1

    y_score, y_true = y_score.flatten(), y_true.flatten()
    metrics = RecEvaluator.evaluate_all(y_true=y_true, y_score=y_score)

    return metrics.dict()


def train(
    pretrained: str = "bert-base-uncased",
    npratio: int = 4,
    history_size: int = 5,
    batch_size: int = 8,
    epochs: int = 5,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.0,
    max_len: int = 16,
    device: torch.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu"),
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
        # metric_for_best_model="f1",
        # load_best_model_at_end=True,
        # evaluation_strategy="no",
        # save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=epochs,
        remove_unused_columns=False,
        report_to="none",
    )
    logging.info(training_args.device)

    nrms_net.set_mode("train")
    trainer = Trainer(
        model=nrms_net,
        compute_metrics=compute_rec_metrics,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)], # TODO:
    )

    trainer.train()

    trainer.evaluate(eval_dataset)


if __name__ == "__main__":
    train()
