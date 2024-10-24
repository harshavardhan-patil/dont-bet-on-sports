from pathlib import Path

from loguru import logger
from src.config import PROCESSED_DATA_DIR
from src.features import prepare_tt_data
from src.modeling.train import train_random_forests, train_gbt, train_support_vectors, train_nn

#generates features, trains models, generates reports
def main():
    #create train and test features
    prepare_tt_data(True)
    prepare_tt_data(False)

    #train all models
    train_random_forests()
    train_support_vectors()
    train_gbt()
    train_nn()


if __name__ == "__main__":
    main()
