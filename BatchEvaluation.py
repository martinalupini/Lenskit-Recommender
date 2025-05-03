from lenskit.als import BiasedMFScorer, ImplicitMFScorer
from lenskit.batch import recommend
from lenskit.data import ItemListCollection, UserIDKey
from lenskit.metrics import NDCG, RecipRank, RunAnalysis
from lenskit.pipeline import topn_pipeline
from lenskit.splitting import SampleFrac, crossfold_users
from lenskit.basic import BiasScorer
from lenskit.knn import UserKNNScorer, ItemKNNScorer
from lenskit.funksvd import FunkSVDScorer
from lenskit.sklearn.svd import BiasedSVDScorer
from  lenskit.basic.random import RandomSelector
import seaborn as sns
import matplotlib.pyplot as plt



def batch_evaluation(ml100k, small=False):
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

    metrics_table = results.list_metrics().groupby("model")[["NDCG", "RecipRank"]].mean()

    # Salva su CSV o LaTeX se serve per report
    if small:
        path = "files/metrics_small.csv"
        path_NDCG = "plots/plotNDCG_small.png"
        path_RR = "plots/plotRR_small.png"
    else:
        path = "files/metrics_2.csv"
        path_NDCG = "plots/plotNDCG.png"
        path_RR = "plots/plotRR.png"
    metrics_table.to_csv(path)

    sns.catplot(results.list_metrics().reset_index(), x="model", y="NDCG", kind="bar", aspect=2)
    plt.savefig(path_NDCG)
    sns.catplot(results.list_metrics().reset_index(), x="model", y="RecipRank", kind="bar", aspect=2)
    plt.savefig(path_RR)



def random_predictor(ml100k):

    try:
        model = RandomSelector()
        pipe = topn_pipeline(model)
        # test data is organized by user
        all_test = ItemListCollection(UserIDKey)

        for split in crossfold_users(ml100k, 5, SampleFrac(0.2)):
            # collect the test data
            all_test.add_from(split.test)

            # train the pipeline, cloning first so a fresh pipeline for each split
            fit_als = pipe.clone()
            fit_als.train(split.train)
            # generate recs
            recommend(fit_als, split.test.keys(), 100)
    except RuntimeError as e:
        print(e)
