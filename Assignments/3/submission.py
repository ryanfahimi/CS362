from Cluster import k_means
from make_dataset import create_documents
from loader import compute_homogeneity, compute_completeness
from MyNaiveBayes import MyNaiveBayes

# Demo K-means
data = create_documents(10, 10, 10)
clusters = k_means(2, ["pos", "neg"], data)
homogeneity = compute_homogeneity(clusters)
completeness = compute_completeness(clusters, ["pos", "neg"])
print(f"Homogeneity: {homogeneity}")
print(f"Completeness: {completeness}")

# Demo Naive Bayes
nb = MyNaiveBayes("breast-cancer.data")
f1_scores = nb.five_fold()
print(f"F1 scores: {f1_scores}")
