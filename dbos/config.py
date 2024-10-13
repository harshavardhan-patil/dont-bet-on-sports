from pathlib import Path

from dotenv import load_dotenv
#from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

#URL
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/a3b24f5aa89213ae2f8c914df7b974972beb6f0f/data/games.csv"

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
#logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
DB_DATA_DIR = DATA_DIR / "db"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    #logger.remove(0)
    #logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
