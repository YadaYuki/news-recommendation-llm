import requests
from const.env import SLACK_WEBHOOK_URL


def notify_slack(text: str) -> None:
    requests.post(SLACK_WEBHOOK_URL, json={"text": text})
