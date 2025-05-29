## Code for loading training sets and creating doc_list.
from collections import Counter


## homogeneity: for each cluster, what fraction of the cluster consists of its most common class?
# call this like so:
# result = k_means(2, positive_docs + negative_docs)
# compute_homogeneity(result)
def compute_homogeneity(clusters):
    # h_list will be the homogeneity for each cluster.
    h_list = []
    for cluster in clusters:
        cluster_counts = Counter(member.true_class for member in cluster.members)
        max_count = cluster_counts.most_common(1)[0][1]
        h_list.append(max_count / len(cluster.members))
    return h_list


## completeness: the fraction of elements of each class captured in the largest cluster for that class
# call this like so:
# result = k_means(2, positive_docs + negative_docs)
# compute_completeness(result)
def compute_completeness(clusters, true_classes):
    # c_list will be the homogeneity for each cluster.
    c_list = []
    for true_class in true_classes:
        counts = [
            Counter(member.true_class for member in cluster.members)[true_class]
            for cluster in clusters
        ]
        max_count = max(counts)
        total_count = sum(counts)
        c_list.append(max_count / total_count) if total_count != 0 else c_list.append(0)
    return c_list
