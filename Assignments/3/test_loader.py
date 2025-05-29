import unittest
from Cluster import Cluster, k_means
from make_dataset import create_documents
from Document import Document
from loader import compute_homogeneity, compute_completeness


class TestLoader(unittest.TestCase):
    def setUp(self):
        self.pos1 = Document(true_class="pos")
        self.pos2 = Document(true_class="pos")
        self.pos3 = Document(true_class="pos")
        self.neg1 = Document(true_class="neg")
        self.neg2 = Document(true_class="neg")
        self.neg3 = Document(true_class="neg")
        self.true_classes = ["pos", "neg"]

    def test_homogeneity_all_same(self):
        cluster = Cluster(members=[self.pos1, self.pos2])
        clusters = [cluster]
        result = compute_homogeneity(clusters)
        self.assertEqual(result[0], 1.0)

    def test_homogeneity_mixed(self):
        cluster = Cluster(members=[self.pos1, self.pos2, self.neg1])
        clusters = [cluster]
        result = compute_homogeneity(clusters)
        self.assertAlmostEqual(result[0], 2 / 3, places=5)

    def test_homogeneity_multiple_clusters(self):
        cluster1 = Cluster(members=[self.pos1, self.neg1])
        cluster2 = Cluster(members=[self.neg2])
        clusters = [cluster1, cluster2]
        result = compute_homogeneity(clusters)
        self.assertAlmostEqual(result[0], 0.5, places=5)
        self.assertAlmostEqual(result[1], 1.0, places=5)

    def test_completeness_all_same(self):
        cluster = Cluster(members=[self.pos1, self.pos2])
        clusters = [cluster]
        result = compute_completeness(clusters, self.true_classes)
        self.assertEqual(result[0], 1.0)

    def test_completeness_mixed(self):
        cluster = Cluster(members=[self.pos1, self.pos2, self.neg1])
        clusters = [cluster]
        result = compute_completeness(clusters, self.true_classes)
        self.assertEqual(result[0], 1.0)

    def test_completeness_multiple_clusters(self):
        cluster1 = Cluster(members=[self.pos1, self.pos2, self.neg1])
        cluster2 = Cluster(members=[self.pos3])
        clusters = [cluster1, cluster2]
        result = compute_completeness(clusters, self.true_classes)
        self.assertAlmostEqual(result[0], 2 / 3, places=5)
        self.assertEqual(result[1], 1.0)

    def test_workflow(self):
        data = create_documents(10, 10, 10)
        clusters = k_means(2, ["pos", "neg"], data)
        homogeneity = compute_homogeneity(clusters)
        completeness = compute_completeness(clusters, self.true_classes)
        self.assertEqual(len(homogeneity), 2)
        self.assertEqual(len(completeness), 2)


if __name__ == "__main__":
    unittest.main()
