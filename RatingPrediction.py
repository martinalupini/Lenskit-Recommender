from lenskit.als import BiasedMFScorer, ImplicitMFScorer
from lenskit.pipeline import topn_pipeline, predict_pipeline
from lenskit.basic import BiasScorer
from lenskit.knn import UserKNNScorer, ItemKNNScorer
from lenskit.funksvd import FunkSVDScorer
from lenskit.sklearn.svd import BiasedSVDScorer
from lenskit.operations import predict
import pandas as pd


def rating_prediction(ml100k):

    model_BiasScorer = BiasScorer(damping=5)
    model_UserKNNScorer = UserKNNScorer(max_nbrs=25)
    model_ItemKNNScorer = ItemKNNScorer(max_nbrs=25)
    model_BiasedMFScorer = BiasedMFScorer(embedding_size=40)
    model_ImplicitMFScorer = ImplicitMFScorer(embedding_size=40)
    model_FunkSVDScorer = FunkSVDScorer(features=40)
    model_BiasedSVDScorer = BiasedSVDScorer(embedding_size=40)

    pipe_BiasScorer = predict_pipeline(model_BiasScorer)
    pipe_UserKNNScorer = predict_pipeline(model_UserKNNScorer)
    pipe_ItemKNNScorer = predict_pipeline(model_ItemKNNScorer)
    pipe_BiasedMFScorer = predict_pipeline(BiasedMFScorer)
    pipe_ImplicitMFScorer = predict_pipeline(ImplicitMFScorer)
    pipe_FunkSVDScorer = predict_pipeline(FunkSVDScorer)
    pipe_BiasedSVDScorer = predict_pipeline(BiasedSVDScorer)

    pipe_BiasScorer.train(ml100k)
    pipe_UserKNNScorer.train(ml100k)
    pipe_ItemKNNScorer.train(ml100k)
    pipe_BiasedMFScorer.train(ml100k)
    pipe_ImplicitMFScorer.train(ml100k)
    pipe_FunkSVDScorer.train(ml100k)
    pipe_BiasedSVDScorer.train(ml100k)

    i = 0
    predictions = []
    while i < 3:
        prediction_BiasScorer = predict(pipe_BiasScorer, 196, [302]).to_df()
        prediction_UserKNNScorer = predict(pipe_UserKNNScorer, 196, [302]).to_df()
        prediction_ItemKNNScorer = predict(pipe_ItemKNNScorer, 196, [302]).to_df()
        prediction_BiasedMFScorer = predict(pipe_BiasedMFScorer, 196, [302]).to_df()
        prediction_ImplicitMFScorer = predict(pipe_ImplicitMFScorer, 196, [302]).to_df()
        prediction_FunkSVDScorer = predict(pipe_FunkSVDScorer, 196, [302]).to_df()
        prediction_BiasedSVDScorer = predict(pipe_BiasedSVDScorer, 196, [302]).to_df()

        # Estrai lo score
        score_BiasScorer = prediction_BiasScorer.loc[0, 'score']
        score_UserKNNScorer = prediction_UserKNNScorer.loc[0, 'score']
        score_ItemKNNScorer = prediction_ItemKNNScorer.loc[0, 'score']
        score_BiasedMFScorer = prediction_BiasedMFScorer.loc[0, 'score']
        score_ImplicitMFScorer = prediction_ImplicitMFScorer.loc[0, 'score']
        score_FunkSVDScorer = prediction_FunkSVDScorer.loc[0, 'score']
        score_BiasedSVDScorer = prediction_BiasedSVDScorer.loc[0, 'score']

        # Raccogli i risultati in un dizionario
        result = {
            "Iteration": i + 1,  # Iterazione corrente
            "BiasScorer": score_BiasScorer,
            "UserKNNScorer": score_UserKNNScorer,
            "ItemKNNScorer": score_ItemKNNScorer,
            "BiasedMFScorer": score_BiasedMFScorer,
            "ImplicitMFScorer": score_ImplicitMFScorer,
            "FunkSVDScorer": score_FunkSVDScorer,
            "BiasedSVDScorer": score_BiasedSVDScorer
        }

        predictions.append(result)

        i += 1

    # Converti la lista di dizionari in un DataFrame
    df_predictions = pd.DataFrame(predictions)

    # Salva il DataFrame in un file CSV
    df_predictions.to_csv("files/predictions.csv", index=False)