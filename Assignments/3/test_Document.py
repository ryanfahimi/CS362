import unittest
from math import sqrt
from Document import Document, euclidean_distance, cosine_similarity


class TestDocument(unittest.TestCase):
    def setUp(self):
        self.doc = Document(true_class="pos")

    def test_init(self):
        self.assertEqual(self.doc.true_class, "pos")
        self.assertEqual(self.doc.tokens["nonexistent"], 0)

    def test_add_tokens(self):
        self.doc.add_tokens(["cat", "dog", "fish", "cat"])
        self.assertEqual(self.doc.tokens["cat"], 2)
        self.assertEqual(self.doc.tokens["dog"], 1)
        self.assertEqual(self.doc.tokens["fish"], 1)
        self.assertEqual(self.doc.tokens["elephant"], 0)


class TestDistance(unittest.TestCase):
    def setUp(self):
        # Create common documents for distance tests.
        self.d1 = Document(true_class="pos")
        self.d1.add_tokens(["cat", "dog", "fish"])

        self.d2 = Document(true_class="pos")
        self.d2.add_tokens(["cat", "dog", "fish"])

        self.d3 = Document(true_class="pos")
        self.d3.add_tokens(["cat", "bunny", "fish"])

        self.d4 = Document(true_class="pos")
        self.d4.add_tokens(["bunny", "mouse", "elephant"])

    def test_euclidean_distance_identical(self):
        self.assertEqual(euclidean_distance(self.d1, self.d2), 0)

    def test_euclidean_distance_disjoint(self):
        # Each token appears only in one document: sum of squares = 6 => sqrt(6).
        self.assertAlmostEqual(euclidean_distance(self.d1, self.d4), sqrt(6), places=5)

    def test_euclidean_distance_partial_overlap(self):
        # For the union {"cat", "dog", "fish", "bunny"}:
        # Difference for "cat": 0, "fish": 0, "dog": (1-0)^2 = 1, "bunny": (0-1)^2 = 1.
        # Sum = 2 => distance = sqrt(2)
        self.assertAlmostEqual(euclidean_distance(self.d1, self.d3), sqrt(2), places=5)

    def test_cosine_similarity_identical(self):
        self.assertAlmostEqual(cosine_similarity(self.d1, self.d2), 1.0, places=5)

    def test_cosine_similarity_disjoint(self):
        self.assertAlmostEqual(cosine_similarity(self.d2, self.d4), 0.0, places=5)

    def test_cosine_similarity_partial_overlap(self):
        # For the union {"cat", "dog", "fish", "bunny"}:
        # numerator = 2, denominator = sqrt(3 * 3) = 3.
        expected = 2 / 3
        self.assertAlmostEqual(cosine_similarity(self.d1, self.d3), expected, places=5)


if __name__ == "__main__":
    unittest.main()
