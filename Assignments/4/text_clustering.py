# derived from https://scikit-learn.org/1.5/auto_examples/text/plot_document_clustering.html
# # Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
# #         Lars Buitinck
# #         Olivier Grisel <olivier.grisel@ensta.org>
# #         Arturo Amor <david-arturo.amor-quiroz@inria.fr>
# # License: BSD 3 clause

import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from collections import defaultdict
from time import time
from sklearn import metrics

## fit_and_evaluate takes as input:
# km: a KMeans object
# X: a list of vectors
# name: an optional name to give the run
# n_runs: how many times do we want to run?

def fit_and_evaluate(km, X, name=None, n_runs=5, labels=""):
    name = km.__class__.__name__ if name is None else name

    train_times = []
    scores = defaultdict(list)
    for seed in range(n_runs):
        km.set_params(random_state=seed)
        t0 = time()
        km.fit(X)
        train_times.append(time() - t0)
        scores["Homogeneity"].append(metrics.homogeneity_score(labels, km.labels_))
        scores["Completeness"].append(metrics.completeness_score(labels, km.labels_))
        scores["V-measure"].append(metrics.v_measure_score(labels, km.labels_))
        scores["Adjusted Rand-Index"].append(
            metrics.adjusted_rand_score(labels, km.labels_)
        )
        scores["Silhouette Coefficient"].append(
            metrics.silhouette_score(X, km.labels_, sample_size=2000)
        )
    train_times = np.asarray(train_times)

    print(f"clustering done in {train_times.mean():.2f} ± {train_times.std():.2f} s ")
    evaluation = {
        "estimator": name,
        "train_time": train_times.mean(),
    }
    evaluation_std = {
        "estimator": name,
        "train_time": train_times.std(),
    }
    for score_name, score_values in scores.items():
        mean_score, std_score = np.mean(score_values), np.std(score_values)
        print(f"{score_name}: {mean_score:.3f} ± {std_score:.3f}")
        evaluation[score_name] = mean_score
        evaluation_std[score_name] = std_score

    return evaluation, evaluation_std

## helper function to get the dataset.
def fetch_data() :
    categories = [
        "alt.atheism",
        "talk.religion.misc",
        "comp.graphics",
        "sci.space",
        # 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
        # 'rec.autos', 'alt.atheism', 'sci.space', 'talk.politics.mideast'
    ]

    dataset = fetch_20newsgroups(
        remove=("headers", "footers", "quotes"),
        subset="all",
        categories=categories,
        shuffle=True,
        random_state=42)

    labels = dataset.target
    unique_labels, category_sizes = np.unique(labels, return_counts=True)
    true_k = unique_labels.shape[0]

    print(f"{len(dataset.data)} documents - {true_k} categories")

    return dataset, labels, unique_labels, category_sizes, true_k

def run_tfidf_cluster() :
    dataset, labels, unique_labels, category_sizes, true_k = fetch_data()

    vectorizer = TfidfVectorizer(
        max_df=0.5,
        min_df=5,
        stop_words="english")
    t0 = time()
    X_tfidf = vectorizer.fit_transform(dataset.data)

    print(f"vectorization done in {time() - t0:.3f} s")
    print(f"n_samples: {X_tfidf.shape[0]}, n_features: {X_tfidf.shape[1]}")

    for seed in range(5):
        kmeans = KMeans(
            n_clusters=true_k,
            max_iter=100,
            n_init=5,
            random_state=seed,
        )
        fit_and_evaluate(kmeans, X_tfidf, name="KMeans tf-idf vectors", labels=labels)
        cluster_ids, cluster_sizes = np.unique(kmeans.labels_, return_counts=True)
        print(f"Number of elements assigned to each cluster: {cluster_sizes}")
    print()
    print(
        "True number of documents in each category according to the class labels: "
        f"{category_sizes}"
    )

def run_vanilla_cluster() :
    dataset, labels, unique_labels, category_sizes, true_k = fetch_data()
    t0 = time()
    vectorizer = HashingVectorizer(n_features=10000,norm=None, stop_words='english')
    X_vanilla = vectorizer.fit_transform(dataset.data)

    print(f"vectorization done in {time() - t0:.3f} s")
    print(f"n_samples: {X_vanilla.shape[0]}, n_features: {X_vanilla.shape[1]}")

    for seed in range(5):
        kmeans = KMeans(
            n_clusters=true_k,
            max_iter=100,
            n_init=5,
            random_state=seed,
        )
        fit_and_evaluate(kmeans, X_vanilla, name="KMeans with counts for vectors", labels=labels)
        cluster_ids, cluster_sizes = np.unique(kmeans.labels_, return_counts=True)
        print(f"Number of elements assigned to each cluster: {cluster_sizes}")
    print()
    print(
        "True number of documents in each category according to the class labels: "
        f"{category_sizes}"
    )

def run_lsa_cluster() :
    dataset, labels, unique_labels, category_sizes, true_k = fetch_data()
    t0 = time()

    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.5,
        min_df=5,
        stop_words="english",)
    t0 = time()
    X_tfidf = tfidf_vectorizer.fit_transform(dataset.data)

    lsa_vectorizer = TruncatedSVD(n_components=100)
    X_lsa = lsa_vectorizer.fit_transform(X_tfidf)

    print(f"vectorization done in {time() - t0:.3f} s")
    print(f"n_samples: {X_lsa.shape[0]}, n_features: {X_lsa.shape[1]}")

    for seed in range(5):
        kmeans = KMeans(
            n_clusters=true_k,
            max_iter=100,
            n_init=5,
            random_state=seed,
        )
        fit_and_evaluate(kmeans, X_lsa, name="KMeans with Latent Semantic Indexing", labels=labels)
        cluster_ids, cluster_sizes = np.unique(kmeans.labels_, return_counts=True)
        print(f"Number of elements assigned to each cluster: {cluster_sizes}")
    print()
    print(
        "True number of documents in each category according to the class labels: "
        f"{category_sizes}"
    )





if __name__ == "__main__" :
    run_tfidf_cluster()
    run_vanilla_cluster()
    run_lsa_cluster()




