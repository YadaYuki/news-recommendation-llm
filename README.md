<div align="center">
    <img height=200 src="./.github/images/news-logo.png" alt="News Contents on Smartphone">
</div>

<h1 align="center">News Recommendation 🌎 </h1>
<p align="center"><strong>Pretrained Large Language Model Based News Recommendation using Python / PyTorch 🚀 </strong></p>

## Overview

- Implementation of Pretrained Large Language Model Based News Recommendation using Python / PyTorch.
- We adopted **Neural News Recommendation with Multi-Head Self-Attention(NRMS)**[], known for its high performance among neural news recommendation methods, as our model.
- We are using language models such as **BERT**[] as the backbone to obtain embedding vectors for news content.

## Project Structure

The project structure is as below.

```bash
$ tree -L 2
├── README.md
├── dataset/
│   └── download_mind.py
├── pyproject.toml
├── requirements-dev.lock
├── requirements.lock
├── src/
│   ├── config/
│   ├── const/
│   ├── evaluation/
│   ├── experiment/
│   ├── mind/
│   ├── recommendation/
│   │   └── nrms/
│   │       ├── AdditiveAttention.py
│   │       ├── NRMS.py
│   │       ├── PLMBasedNewsEncoder.py
│   │       ├── UserEncoder.py
│   │       ├── __init__.py
│   └── utils/
└── test/
    ├── evaluation/
    ├── mind/
    └── recommendation/
```

## Preparation

### Prerequisites

- [Rye](https://rye-up.com/)
- Python 3.11.3
- PyTorch 2.0.1
- transformers 4.30.2

### Setup

At first, create python virtualenv & install dependencies by running

```
$ rye sync
```

If you successfully created a virtual environment, a `.venv/` folder should be created at the project root.

Then, please set `PYTHONPATH` by runnning

```
$ export PYTHONPATH=$(pwd)/src:$PYTHONPATH
```

### Download Microsoft News Dataset (MIND)

We use **[MIND (Microsoft News Dataset)](https://msnews.github.io/)** [3] dataset for training and validating the news recommendation model. You can download them by executing [dataset/download_mind.py](https://github.com/YadaYuki/news-recommendation-llm/blob/main/dataset/download_mind.py).

```
$ rye run python ./dataset/download_mind.py
```

By executing [dataset/download_mind.py](https://github.com/YadaYuki/news-recommendation-llm/blob/main/dataset/download_mind.py), the MIND dataset will be downloaded from an external site and then extracted.
If you successfully executed, `dataset` folder will be structured as follows:

```
./dataset/
├── download_mind.py
└── mind
    ├── large
    │   ├── test
    │   ├── train
    │   └── val
    ├── small
    │   ├── train
    │   └── val
    └── zip
        ├── MINDlarge_dev.zip
        ├── MINDlarge_test.zip
        ├── MINDlarge_train.zip
        ├── MINDsmall_dev.zip
        └── MINDsmall_train.zip
```

## Experiment

### Fine-tune a news recommendation model

### Trained Model

### Model Performance

## Reference

[1] NRMS
[2] BERT
[3] MIND

## License
