from pydantic import BaseModel


class RecMetrics(BaseModel):
    ndcg_at_10: float
    ndcg_at_5: float
    auc: float
    mrr: float


class RecEvaluator:
    @classmethod
    def evaluate_all(cls) -> RecMetrics:
        return RecMetrics(**{})

    @classmethod
    def __dcg_at_k(cls) -> float:
        return 0.0

    @classmethod
    def ndcg_at_k(cls) -> float:
        return 0.0

    @classmethod
    def auc(cls) -> float:
        return 0.0

    @classmethod
    def mrr(cls) -> float:
        return 0.0
