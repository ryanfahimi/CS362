import unittest
import random
from Cluster import Cluster, k_means
from Document import Document, cosine_similarity


class TestCluster(unittest.TestCase):
    def setUp(self):
        self.d1 = Document(true_class="pos")
        self.d1.add_tokens(["cat", "dog", "fish"])

        self.d2 = Document(true_class="pos")
        self.d2.add_tokens(["cat", "dog", "fish"])

        self.d3 = Document(true_class="neg")
        self.d3.add_tokens(["bunny", "lizard", "octopus"])

    def test_calculate_centroid_two_members(self):
        cluster = Cluster(members=[self.d1, self.d2])
        centroid = cluster.calculate_centroid()
        self.assertAlmostEqual(centroid.tokens["cat"], 1.0, places=5)
        self.assertAlmostEqual(centroid.tokens["dog"], 1.0, places=5)
        self.assertAlmostEqual(centroid.tokens["fish"], 1.0, places=5)
        self.assertEqual(centroid.tokens.get("bunny", 0), 0)

    def test_calculate_centroid_three_members(self):
        cluster = Cluster(members=[self.d1, self.d2, self.d3])
        centroid = cluster.calculate_centroid()
        self.assertAlmostEqual(centroid.tokens["cat"], 2 / 3, places=5)
        self.assertAlmostEqual(centroid.tokens["dog"], 2 / 3, places=5)
        self.assertAlmostEqual(centroid.tokens["fish"], 2 / 3, places=5)
        self.assertAlmostEqual(centroid.tokens["bunny"], 1 / 3, places=5)
        self.assertAlmostEqual(centroid.tokens["lizard"], 1 / 3, places=5)
        self.assertAlmostEqual(centroid.tokens["octopus"], 1 / 3, places=5)

    def test_k_means(self):
        random.seed(42)
        data = [self.d1, self.d2, self.d3]
        clusters = k_means(2, ["pos", "neg"], data)
        self.assertEqual(len(clusters), 2)
        for cluster in clusters:
            if any(document.tokens.get("bunny", 0) for document in cluster.members):
                self.assertEqual(len(cluster.members), 1)
            else:
                self.assertEqual(len(cluster.members), 2)


if __name__ == "__main__":
    unittest.main()
