from datetime import datetime
from pathlib import Path


def generate_folder_name_with_timestamp(path_prefix: Path, timestamp: datetime = datetime.now()) -> Path:
    date = Path(timestamp.strftime("%Y-%m-%d"))
    time = Path(timestamp.strftime("%H-%M-%S"))
    return path_prefix / date / time
