from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import requests

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, GAMES_URL

#todo: automate fetching updated dataset and appending to existing post feature pieplines
def main(
    input_path: Path = RAW_DATA_DIR / "games.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
):
    response = requests.get(GAMES_URL)
    if response.status_code == 200:
        input_path.write_bytes(response.content)
        logger.info("Data downloaded successfully")
    else:
        logger.error("Failed to download data. Status code:", response.status_code)

if __name__ == "__main__":
    main()
