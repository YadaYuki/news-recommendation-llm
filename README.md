<div align="center">
    <img height=200 src="./.github/images/news-logo.png" alt="News Contents on Smartphone">
</div>

<h1 align="center">News Recommendation using LLM 🌎 </h1>
<p align="center"><strong>Pre-trained Large Language Model Based News Recommendation using Python / PyTorch 🚀 </strong></p>

## Overview

- Implementation of Pretrained Large Language Model Based News Recommendation using Python / PyTorch.
- We adopted **Neural News Recommendation with Multi-Head Self-Attention(NRMS)**, known for its high performance among neural news recommendation methods, as our model.
- We are using language models such as **BERT** and **DistilBERT** as the backbone to obtain embedding vectors for news content.

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

We use **[MIND (Microsoft News Dataset)](https://msnews.github.io/)** dataset for training and validating the news recommendation model. You can download them by executing [dataset/download_mind.py](https://github.com/YadaYuki/news-recommendation-llm/blob/main/dataset/download_mind.py).

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

### Fine Tune a model

If you execute `src/experiments/train.py`, the news recommendation model will be finetuned on the **MIND small dataset**.
Hyperparameters can be specified from the arguments.

```bash
$ rye run python src/experiments/train.py -m \
    random_seed = 42 \
    pretrained = "distilbert-base-uncased" \
    npratio = 4 \
    history_size = 50 \
    batch_size = 16 \
    gradient_accumulation_steps = 8 \
    epochs = 3 \
    learning_rate = 1e-4 \
    weight_decay = 0.0 \
    max_len = 30 \
```

You can see the default values for each hyperparameter in [src/config/config.py](https://github.com/YadaYuki/news-recommendation-llm/blob/feat/add-readme/src/config/config.py#L1-L23). If you simply execute `rye run python train.py`, fine-tuning will start based on the default values.

### Model Performance

We ran the fine-tuning code on Single GPU (V100 x 1). Then, evaluated on validation set of MIND Small Dataset. Additionally, as a point of comparison, we implemented **random** recommendations ([`src/experiments/evaluate_random.py`](https://github.com/YadaYuki/news-recommendation-llm/blob/feat/add-readme/src/experiment/evaluate_random.py) ) and evaluated.

#### Experimental Result

|         Model          |  AUC  |  MRR  | nDCG@5 | nDCG@10 | Time to Train |
| :--------------------: | :---: | :---: | :----: | :-----: | :-----------: |
| Random Recommendation  | 0.500 | 0.201 | 0.203  |  0.267  |       -       |
| NRMS + DistilBERT-base | 0.674 | 0.297 | 0.322  |  0.387  |    15.0 h     |
|    NRMS + BERT-base    | 0.689 | 0.306 | 0.336  |  0.400  |    28.5 h     |

### Trained Model

To make it easy to try inference and evaluation, we have publicly released the trained model.
Here are the links.

|         Model          |                                                Link                                                |
| :--------------------: | :------------------------------------------------------------------------------------------------: |
| NRMS + DistilBERT-base | [Google Drive](https://drive.google.com/file/d/1cw9WQSOVYJdYJCuIrSmU8odV2nsmith5/view?usp=sharing) |
|    NRMS + BERT-base    | [Google Drive](https://drive.google.com/file/d/1ARiUgSVwcDFopFoIusp2MGQzwTMncOFf/view?usp=sharing) |

You can try it with the following script.

```python
import torch
from recommendation.nrms import NRMS, PLMBasedNewsEncoder, UserEncoder

loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss()
pretrained = "distilbert-base-uncased"
news_encoder = PLMBasedNewsEncoder(pretrained)
user_encoder = UserEncoder(hidden_size=hidden_size)
nrms_net = NRMS(news_encoder=news_encoder, user_encoder=user_encoder, hidden_size=hidden_size, loss_fn=loss_fn).to(
    device, dtype=torch.bfloat16
)
path_to_model = {path to trained NRMS-DistilBERT Model}
nrms_net.load_state_dict(torch.load(path_to_model))
```

## Citation

```
@misc{
  yuki-yada-news-recommendation-llm,
  author = {Yuki Yada},
  title = {News Recommendation using LLM},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/YadaYuki/news-recommendation-llm}}
}
```

## Reference

[1] **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**
Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K.
https://aclanthology.org/N19-1423

[2] **DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter**
Sanh, V., Debut, L., Chaumond, J., & Wolf, T.
https://arxiv.org/abs/1910.01108

[3] **Neural News Recommendation with Multi-Head Self-Attention**
Wu, C., Wu, F., Ge, S., Qi, T., Huang, Y., & Xie, X.
https://aclanthology.org/D19-1671

[4] **Empowering News Recommendation with Pre-Trained Language Models**
Wu, C., Wu, F., Qi, T., & Huang, Y.
https://doi.org/10.1145/3404835.3463069

[5] **MIND: A Large-scale Dataset for News Recommendation**
Wu, F., Qiao, Y., Chen, J.-H., Wu, C., Qi, T., Lian, J., Liu, D., Xie, X., Gao, J., Wu, W., & Zhou, M.
https://aclanthology.org/2020.acl-main.331

## Using This Software in Your Work

We warmly welcome the use of this software in academic, research, or educational settings. If you decide to use our software in your paper or project, we would appreciate it if you could inform us. This helps us understand the impact and reach of our software and can guide future development and improvements.

Please contact us at: yada.yuki@fuji.waseda.jp

### Citation

If you refer to or use our software in your research, please cite it as follows:
```
@misc{yuki-yada-news-rec,
  author = {Yuki Yada},
  title = {News Recommendation using PLMs},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/YadaYuki/news-recommendation-llm}}
}

```
