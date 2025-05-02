from lenskit.als import BiasedMFScorer, ImplicitMFScorer
from lenskit.batch import recommend
from lenskit.data import ItemListCollection, UserIDKey, load_movielens
from lenskit.metrics import NDCG, RecipRank, RunAnalysis
from lenskit.pipeline import topn_pipeline
from lenskit.splitting import SampleFrac, crossfold_users
from lenskit.basic import BiasScorer
from lenskit.knn import UserKNNScorer, ItemKNNScorer
from lenskit.funksvd import FunkSVDScorer
from lenskit.sklearn.svd import BiasedSVDScorer
from lenskit.operations import predict
import seaborn as sns
from pyprojroot.here import here
#import matplotlib
#matplotlib.use('TkAgg')  # o 'Qt5Agg' se hai Qt installato
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    # Caricamento del file con pandas
    #ml100k = pd.read_csv('dataset/u.data', sep=r'\s+', names=['user', 'item', 'rating', 'timestamp'])

    ml100k = load_movielens(here("dataset/ml-100k.zip"))


    model_BiasScorer = BiasScorer(damping=5)
    model_UserKNNScorer = UserKNNScorer(max_nbrs=25)
    model_ItemKNNScorer = ItemKNNScorer(max_nbrs=25)
    model_BiasedMFScorer = BiasedMFScorer(embedding_size=40)
    model_ImplicitMFScorer = ImplicitMFScorer(embedding_size=40)
    model_FunkSVDScorer = FunkSVDScorer(features=40)
    model_BiasedSVDScorer = BiasedSVDScorer(embedding_size=40)

    pipe_BiasScorer = topn_pipeline(model_BiasScorer)
    pipe_UserKNNScorer = topn_pipeline(model_UserKNNScorer)
    pipe_ItemKNNScorer = topn_pipeline(model_ItemKNNScorer)
    pipe_BiasedMFScorer = topn_pipeline(BiasedMFScorer)
    pipe_ImplicitMFScorer = topn_pipeline(ImplicitMFScorer)
    pipe_FunkSVDScorer = topn_pipeline(FunkSVDScorer)
    pipe_BiasedSVDScorer = topn_pipeline(BiasedSVDScorer)

    """
    # test data is organized by user
    all_test = ItemListCollection(UserIDKey)
    # recommendations will be organized by model and user ID
    all_recs = ItemListCollection(["model", "user_id"])

    for split in crossfold_users(ml100k, 5, SampleFrac(0.2)):
        # collect the test data
        all_test.add_from(split.test)

        # train the pipeline, cloning first so a fresh pipeline for each split
        fit_als = pipe_BiasScorer.clone()
        fit_als.train(split.train)
        # generate recs
        als_recs = recommend(fit_als, split.test.keys(), 100)
        all_recs.add_from(als_recs, model="BiasScorer")

        # do the same for item-item
        fit_ii = pipe_UserKNNScorer.clone()
        fit_ii.train(split.train)
        ii_recs = recommend(fit_ii, split.test.keys(), 100)
        all_recs.add_from(ii_recs, model="UserKNNScorer")

        fit_uu = pipe_ItemKNNScorer.clone()
        fit_uu.train(split.train)
        uu_recs = recommend(fit_uu, split.test.keys(), 100)
        all_recs.add_from(uu_recs, model="ItemKNNScorer")

        fit_biasMF = pipe_BiasedMFScorer.clone()
        fit_biasMF.train(split.train)
        biasMF_recs = recommend(fit_biasMF, split.test.keys(), 100)
        all_recs.add_from(biasMF_recs, model="BiasedMFScorer")

        fit_ImplicitMFScorer = pipe_ImplicitMFScorer.clone()
        fit_ImplicitMFScorer.train(split.train)
        ImplicitMFScorer_recs = recommend(fit_ImplicitMFScorer, split.test.keys(), 100)
        all_recs.add_from(ImplicitMFScorer_recs, model="ImplicitMFScorer")

        fit_FunkSVDScorer = pipe_FunkSVDScorer.clone()
        fit_FunkSVDScorer.train(split.train)
        FunkSVDScorer_recs = recommend(fit_FunkSVDScorer, split.test.keys(), 100)
        all_recs.add_from(FunkSVDScorer_recs, model="FunkSVDScorer")

        fit_BiasedSVDScorer = pipe_BiasedSVDScorer.clone()
        fit_BiasedSVDScorer.train(split.train)
        BiasedSVDScorer_recs = recommend(fit_BiasedSVDScorer, split.test.keys(), 100)
        all_recs.add_from(BiasedSVDScorer_recs, model="BiasedSVDScorer")


    ran = RunAnalysis()
    ran.add_metric(NDCG())
    ran.add_metric(RecipRank())
    results = ran.measure(all_recs, all_test)

    results.list_metrics().groupby("model").mean()

    sns.catplot(results.list_metrics().reset_index(), x="model", y="NDCG", kind="bar", aspect=2)
    plt.savefig("plots/plotNDCG.png")
    sns.catplot(results.list_metrics().reset_index(), x="model", y="RecipRank", kind="bar", aspect=2)
    plt.savefig("plots/plotRR.png")
    """
    #################################################
    pipe_BiasScorer.train(ml100k)
    pipe_UserKNNScorer.train(ml100k)
    pipe_ItemKNNScorer.train(ml100k)
    pipe_BiasedMFScorer.train(ml100k)
    pipe_ImplicitMFScorer.train(ml100k)
    pipe_FunkSVDScorer.train(ml100k)
    pipe_BiasedSVDScorer.train(ml100k)

    prediction_BiasScorer = predict(pipe_BiasScorer, 196, [302])
    prediction_UserKNNScorer = predict(pipe_UserKNNScorer, 196, [302])
    prediction_ItemKNNScorer = predict(pipe_ItemKNNScorer, 196, [302])
    prediction_BiasedMFScorer = predict(pipe_BiasedMFScorer, 196, [302])
    prediction_ImplicitMFScorer = predict(pipe_ImplicitMFScorer, 196, [302])
    prediction_FunkSVDScorer = predict(pipe_FunkSVDScorer, 196, [302])
    prediction_BiasedSVDScorer = predict(pipe_BiasedSVDScorer, 196, [302])
