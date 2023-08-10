import hashlib
import inspect
import json
import pickle
from pathlib import Path
from typing import Callable

import pandas as pd
import polars as pl
from const.path import CACHE_DIR
from utils.logger import logging


def _cache_dataframe(fn: Callable) -> Callable:
    def read_df_function_wrapper(*args: tuple, **kwargs: dict) -> pl.DataFrame:
        # inspect **kwargs
        bound = inspect.signature(fn).bind(*args, **kwargs)
        bound.apply_defaults()

        d = bound.arguments
        d["function_name"] = fn.__name__
        d["path_to_tsv"] = str(bound.arguments["path_to_tsv"])

        # if file exist in cache path, then load & return it.
        cache_filename = hashlib.sha256(json.dumps(d).encode()).hexdigest()
        cache_path = CACHE_DIR / f"{cache_filename}.pth"
        if cache_path.exists() and (not d["clear_cache"]):
            with open(cache_path, "rb") as f:
                df = pickle.load(f)
            return df

        df = fn(*args, **kwargs)

        cache_path.parent.mkdir(parents=True, exist_ok=True)

        with open(cache_path, "wb") as f:
            pickle.dump(df, f)

        return df

    return read_df_function_wrapper


@_cache_dataframe
def read_news_df(path_to_tsv: Path, has_entities: bool = False, clear_cache: bool = False) -> pl.DataFrame:
    # FIXME:
    # pl.read_csvを直接実行すると、行が欠損するため、pandasでtsvを読み取り、polarsのDataFrameに変換する
    news_df = pd.read_csv(path_to_tsv, sep="\t", encoding="utf8", header=None)
    news_df.columns = [
        "news_id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities",
    ]
    news_df = pl.from_dataframe(news_df)
    if has_entities:
        return news_df
    return news_df.drop("title_entities", "abstract_entities")


@_cache_dataframe
def read_behavior_df(path_to_tsv: Path, clear_cache: bool = False) -> pl.DataFrame:
    behavior_df = pl.read_csv(path_to_tsv, separator="\t", encoding="utf8-lossy", has_header=False)
    behavior_df = behavior_df.rename(
        {
            "column_1": "impression_id",
            "column_2": "user_id",
            "column_3": "time",
            "column_4": "history_str",
            "column_5": "impressions_str",
        }
    )
    behavior_df = (
        behavior_df.with_columns((pl.col("impressions_str").str.split(" ")).alias("impression_news_list"))
        .with_columns(
            [
                pl.col("impression_news_list")
                .apply(lambda v: [{"news_id": item.split("-")[0], "clicked": int(item.split("-")[1])} for item in v])
                .alias("impressions")
            ]
        )
        .with_columns([pl.col("history_str").str.split(" ").alias("history")])
        .select(["impression_id", "user_id", "time", "history", "impressions"])
    )
    logging.info(behavior_df[0])
    return behavior_df
