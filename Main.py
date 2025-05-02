from lenskit.data import  load_movielens
from pyprojroot.here import here
import pandas as pd
from BatchEvaluation import *
from RatingPrediction import *


if __name__ == "__main__":
    # Caricamento del file con pandas
    ml100k_small = pd.read_csv('dataset-small/ratings.csv', sep=',', names=['users', 'item', 'rating', 'timestamp'], skiprows=1)

    print(ml100k_small.head())
    ml100k = load_movielens(here("dataset/ml-100k.zip"))


    #batch_evaluation(ml100k, small=False)
    #rating_prediction(ml100k)
    #random_predictor(ml100k)
    batch_evaluation(ml100k_small, small=True)
