import os

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
assert SLACK_WEBHOOK_URL is not None
