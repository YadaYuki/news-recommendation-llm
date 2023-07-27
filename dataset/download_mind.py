from src.const.path import DATASET_DIR
import requests
from tqdm import tqdm
from pydantic import BaseModel
from zipfile import ZipFile
from concurrent.futures import ThreadPoolExecutor, wait

MIND_DATASET_BASE_URL = "https://mind201910small.blob.core.windows.net/release"
MIND_DATASET_DIR = DATASET_DIR / "mind"
MIND_ZIP_DIR = MIND_DATASET_DIR / "zip"


class DataItemType(BaseModel):
    data_size: str
    data_type: str
    zip_filename: str


def download_mind(zip_filename: str) -> None:
    dataset_url = f"{MIND_DATASET_BASE_URL}/{zip_filename}"
    res = requests.get(dataset_url, stream=True)
    KB = 1024
    data_size = int(res.headers.get("content-length", 0))
    progress_bar = tqdm(total=data_size, unit="iB", unit_scale=True, desc=f"[{zip_filename}]")

    with open(MIND_ZIP_DIR / zip_filename, "wb") as file:
        for chunk in res.iter_content(KB):
            progress_bar.update(len(chunk))
            file.write(chunk)
    progress_bar.close()


def download_mind_dataset() -> None:
    """
    1. Download Microsoft News Dataset.
    """
    data_item_list: list[DataItemType] = [
        DataItemType(**{"data_size": "small", "data_type": "train", "zip_filename": "MINDsmall_train.zip"}),
        DataItemType(**{"data_size": "small", "data_type": "val", "zip_filename": "MINDsmall_dev.zip"}),
        DataItemType(**{"data_size": "large", "data_type": "train", "zip_filename": "MINDlarge_train.zip"}),
        DataItemType(**{"data_size": "large", "data_type": "val", "zip_filename": "MINDlarge_dev.zip"}),
        DataItemType(**{"data_size": "large", "data_type": "test", "zip_filename": "MINDlarge_test.zip"}),
    ]

    with ThreadPoolExecutor() as executor:
        res = [executor.submit(download_mind, item.zip_filename) for item in data_item_list]
        wait(res)

    """
    2. Extract zip format Dataset.
    """
    for data_item in data_item_list:
        zip_file_path = MIND_ZIP_DIR / data_item.zip_filename
        extract_dir = MIND_DATASET_DIR / data_item.data_size / data_item.data_type
        extract_dir.mkdir(parents=True, exist_ok=True)
        zf = ZipFile(zip_file_path, "r")
        zf.extractall(extract_dir)
        zf.close()


if __name__ == "__main__":
    download_mind_dataset()
