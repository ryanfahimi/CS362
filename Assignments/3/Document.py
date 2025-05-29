## A representation of a document as a set of tokens.

from collections import defaultdict
from math import sqrt


## true_class is either 'pos' or 'neg'
## tokens is a dict that maps tokens to the number of times they occur
## in the document.
class Document:
    def __init__(self, true_class=None):
        self.true_class = true_class
        self.tokens = defaultdict(lambda: 0)

    def add_tokens(self, token_list):
        for item in token_list:
            self.tokens[item] = self.tokens[item] + 1

    def __repr__(self):
        return f"{self.true_class} {self.tokens}"


# return the distance between two doc_list
def euclidean_distance(d1, d2):
    return sqrt(
        sum(
            [
                (d1.tokens[item] - d2.tokens[item]) ** 2
                for item in d1.tokens.keys() | d2.tokens.keys()
            ]
        )
    )


## You implement this.
def cosine_similarity(d1, d2):
    numerator = sum(
        [
            (d1.tokens[item] * d2.tokens[item])
            for item in d1.tokens.keys() | d2.tokens.keys()
        ]
    )
    denominator = sqrt(sum([d1.tokens[item] ** 2 for item in d1.tokens.keys()])) * sqrt(
        sum([d2.tokens[item] ** 2 for item in d2.tokens.keys()])
    )
    return numerator / denominator
