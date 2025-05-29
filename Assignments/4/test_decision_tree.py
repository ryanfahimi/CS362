import pandas as pd
import unittest
from decision_tree import (
    Node,
    entropy,
    gain,
    select_attribute,
    make_tree,
    classify,
)

class Test(unittest.TestCase):

    def test_entropy(self):
        data = pd.Series(['no', 'no', 'yes', 'yes', 'yes', 'no',
                          'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no'])
        e = entropy(data)
        self.assertAlmostEqual(e, 0.94, places=2)

    def test_entropy_uniform(self):
        # Entropy should be 0 if all values are identical.
        data = pd.Series([1, 1, 1, 1])
        self.assertEqual(entropy(data), 0)

    def test_gain_alternating(self):
        # Test gain on alternating classes with two distinct feature values.
        test_variables = pd.Series([1, 2, 1, 2, 1, 2, 1, 2])
        test_classes = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
        # With alternating classes, the gain should be 1.
        self.assertEqual(gain(test_variables, test_classes), 1)

    def test_gain_uniform(self):
        # When the classifications for each feature value are uniform, the gain should be 0.
        test_variables = pd.Series([1, 2, 1, 2, 1, 2, 1, 2])
        test_classes = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
        self.assertEqual(gain(test_variables, test_classes), 0)

    def test_gain_tennis(self):
        # Use a tennis-like dataset to check a known gain value.
        tennis_variables = pd.Series(['sunny', 'sunny', 'sunny', 'sunny', 'sunny',
                                      'overcast', 'overcast', 'overcast', 'overcast',
                                      'rainy', 'rainy', 'rainy', 'rainy', 'rainy'])
        tennis_classes = pd.Series(['yes', 'yes', 'no', 'no', 'no',
                                    'yes', 'yes', 'yes', 'yes',
                                    'yes', 'yes', 'yes', 'no', 'no'])
        # Expected gain is approximately 0.25.
        self.assertAlmostEqual(gain(tennis_variables, tennis_classes), 0.25, places=2)

    def test_select_attribute_single_feature(self):
        # If only one feature is available, it should be selected regardless of gain.
        data = pd.DataFrame({'A': [1, 2, 1, 2]})
        classifications = pd.Series(['yes', 'no', 'yes', 'no'])
        self.assertEqual(select_attribute(data, classifications), 'A')

    def test_select_attribute_multiple_features(self):
        # Create a DataFrame with multiple features where one clearly provides a better split.
        data = pd.DataFrame({
            'A': ['red', 'red', 'blue', 'blue', 'red', 'blue'],
            'B': ['small', 'large', 'small', 'large', 'small', 'large'],
            'C': [10, 20, 10, 20, 10, 20]
        })
        classifications = pd.Series(['yes', 'yes', 'no', 'no', 'yes', 'no'])
        self.assertEqual(select_attribute(data, classifications), 'A')

    def test_select_attribute_tennis(self):
        data = pd.read_csv('tennis.csv')
        indep_vars = data.drop(['play'], axis=1)
        dep_vars = data['play']
        self.assertEqual(select_attribute(indep_vars, dep_vars), "outlook")

    def test_make_tree_all_same(self):
        # Base Case 1: When all classifications are the same, make_tree should return a leaf node.
        features = pd.DataFrame({'A': [1, 2, 3]})
        classifications = pd.Series(['yes', 'yes', 'yes'])
        attr_dict = {'A': [1, 2, 3]}
        tree = make_tree(features, classifications, zero_r_val='default', attr_dict=attr_dict)
        self.assertTrue(tree.is_leaf())
        self.assertEqual(tree.classification, 'yes')

    def test_make_tree_no_rows(self):
        # Base Case 2: When there are no rows, make_tree should return a leaf node using zero_r_val.
        features = pd.DataFrame({'A': []})
        classifications = pd.Series([], dtype=object)
        attr_dict = {'A': [1, 2, 3]}
        tree = make_tree(features, classifications, zero_r_val='default', attr_dict=attr_dict)
        self.assertTrue(tree.is_leaf())
        self.assertEqual(tree.classification, 'default')

    def test_make_tree_no_columns(self):
        # Base Case 3: When there are no columns, make_tree should return a leaf node using zero_r_val.
        features = pd.DataFrame()
        classifications = pd.Series([1, 2, 3])
        tree = make_tree(features, classifications, zero_r_val='default', attr_dict={})
        self.assertTrue(tree.is_leaf())
        self.assertEqual(tree.classification, 'default')

    def test_make_tree_recursive(self):
        # Recursive Case: Create a dataset that splits clearly on attribute 'A'.
        features = pd.DataFrame({
            'A': ['red', 'red', 'blue', 'blue'],
            'B': ['small', 'large', 'small', 'large']
        })
        classifications = pd.Series(['yes', 'yes', 'no', 'no'])
        attr_dict = {
            'A': ['red', 'blue'],
            'B': ['small', 'large']
        }
        tree = make_tree(features, classifications, zero_r_val='default', attr_dict=attr_dict)
        # Since the gain for 'A' should be higher, the root node should split on 'A'.
        self.assertFalse(tree.is_leaf())
        self.assertEqual(tree.attribute, 'A')
        # For A == 'red', all classifications are 'yes'.
        self.assertTrue(tree.children['red'].is_leaf())
        self.assertEqual(tree.children['red'].classification, 'yes')
        # For A == 'blue', all classifications are 'no'.
        self.assertTrue(tree.children['blue'].is_leaf())
        self.assertEqual(tree.children['blue'].classification, 'no')

    def test_classify_leaf(self):
        # If the tree is a leaf, classify should simply return its classification.
        leaf = Node(classification='yes')
        data = pd.Series({'A': 1, 'B': 2})
        self.assertEqual(classify(leaf, data), 'yes')

    def test_classify(self):
        root = Node(attribute='color')
        root.children = {
            'red': Node(classification='apple'),
            'green': Node(classification='leaf')
        }
        data1 = pd.Series({'color': 'red', 'size': 'small'})
        data2 = pd.Series({'color': 'green', 'size': 'big'})
        self.assertEqual(classify(root, data1), 'apple')
        self.assertEqual(classify(root, data2), 'leaf')


if __name__ == '__main__':
    unittest.main()
