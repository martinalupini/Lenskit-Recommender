from lenskit.data import  load_movielens
from pyprojroot.here import here
from BatchEvaluation import *
from RatingPrediction import *


if __name__ == "__main__":

    ml100k_small = load_movielens(here("dataset-small/ml-latest-small.zip"))
    ml100k = load_movielens(here("dataset/ml-100k.zip"))


    batch_evaluation(ml100k, small=False)
    rating_prediction(ml100k)
    random_predictor(ml100k)
    batch_evaluation(ml100k_small, small=True)
