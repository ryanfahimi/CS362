import random

from Document import Document, cosine_similarity
from loader import compute_homogeneity, compute_completeness
from make_dataset import create_documents


class Cluster:
    ## a cluster is a group of doc_list
    def __init__(self, centroid=None, members=None):
        if centroid:
            self.centroid = centroid
        else:
            self.centroid = Document(true_class="pos")
        if members:
            self.members = members
        else:
            self.members = []

    def __repr__(self):
        return f"{self.centroid} {len(self.members)}"

    ## You do this.
    def calculate_centroid(self):
        if self.members:
            new_centroid = Document()
            for document in self.members:
                for token in document.tokens:
                    new_centroid.tokens[token] += document.tokens[token]
            for token in new_centroid.tokens:
                new_centroid.tokens[token] /= len(self.members)
            self.centroid = new_centroid
        return self.centroid


def k_means(n_clusters, true_classes, data, limit=1000, n_runs=10, fraction_seeded=0.1):
    best_score, best_clusters = -1, None

    for run in range(n_runs):
        if fraction_seeded > 0:
            clusters, remaining_documents = seeded_initialization(
                n_clusters, true_classes, data, fraction_seeded
            )
            assign_documents(clusters, remaining_documents)
        else:
            clusters = random_initialization(n_clusters, data)
            assign_documents(clusters, data)

        for _ in range(limit):
            if not update_clusters(clusters):
                break

        homogeneity = compute_homogeneity(clusters)
        completeness = compute_completeness(clusters, true_classes)
        score = sum(homogeneity + completeness) / (2 * n_clusters)

        if score > best_score:
            best_score, best_clusters = score, clusters

    return best_clusters


def seeded_initialization(n_clusters, true_classes, data, fraction_seeded):
    class_documents = {
        true_class: [document for document in data if document.true_class == true_class]
        for true_class in true_classes
    }
    clusters = [Cluster() for _ in range(n_clusters)]

    for i, true_class in enumerate(true_classes):
        documents = class_documents[true_class]
        num_seeded = int(fraction_seeded * len(documents))
        if num_seeded > 0:
            seeded_documents = random.sample(documents, num_seeded)
            clusters[i].members.extend(seeded_documents)
            for doc in seeded_documents:
                class_documents[true_class].remove(doc)
            clusters[i].calculate_centroid()
        else:
            clusters[i].centroid = random.choice(data)

    remaining_documents = [
        document for documents in class_documents.values() for document in documents
    ]

    return clusters, remaining_documents


def random_initialization(n_clusters, data):
    initial_centroids = random.sample(data, n_clusters)
    clusters = [Cluster(centroid=centroid) for centroid in initial_centroids]
    return clusters


def assign_documents(clusters, data):
    for document in data:
        closest_cluster = max(
            clusters, key=lambda c: cosine_similarity(c.centroid, document)
        )
        closest_cluster.members.append(document)


def update_clusters(clusters):
    done = False
    for i, cluster in enumerate(clusters):
        cluster.calculate_centroid()

    for cluster in clusters:
        for document in cluster.members[:]:
            closest_cluster = max(
                clusters, key=lambda c: cosine_similarity(c.centroid, document)
            )
            if closest_cluster is not cluster:
                closest_cluster.members.append(document)
                cluster.members.remove(document)
                done = True
    return done


def main():
    documents = create_documents(10, 10, 10)
    clusters = k_means(2, ["pos", "neg"], documents)
    print(compute_homogeneity(clusters))
    print(compute_completeness(clusters, ["pos", "neg"]))


if __name__ == "__main__":
    main()
