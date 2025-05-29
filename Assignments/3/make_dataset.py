## generate a simple dataset for clustering.
import os
import pickle
import random
from Document import Document

import nltk
from nltk.corpus import words


def generate_lexicons(lexicon_length, common_fraction=0.2):
    nltk.download("words")
    lexicon = list(set(words.words()))
    pos_lexicon = random.sample(lexicon, lexicon_length)
    common_count = int(common_fraction * lexicon_length)
    unique_count = lexicon_length - common_count
    common_words = random.sample(pos_lexicon, common_count)
    remaining_words = list(set(lexicon) - set(pos_lexicon))
    new_words = random.sample(remaining_words, unique_count)
    neg_lexicon = common_words + new_words
    random.shuffle(neg_lexicon)
    return neg_lexicon, pos_lexicon


def save_lexicons(neg_file, neg_lexicon, pos_file, pos_lexicon):
    with open(pos_file, "wb") as pf:
        pickle.dump(pos_lexicon, pf)
    with open(neg_file, "wb") as nf:
        pickle.dump(neg_lexicon, nf)


def load_lexicons(pos_file="pos_lexicon.pkl", neg_file="neg_lexicon.pkl", length=500):
    with open(pos_file, "rb") as pf:
        pos_lexicon = pickle.load(pf)
    with open(neg_file, "rb") as nf:
        neg_lexicon = pickle.load(nf)
    return neg_lexicon[:length], pos_lexicon[:length]


# n_pos - number of positive doc_list
# n_neg - number of negative doc_list
# length - the length of each document.
def create_tokens(n_pos, n_neg, length):
    neg_lexicon, pos_lexicon = load_lexicons()
    pos_docs = []
    neg_docs = []
    for i in range(n_pos):
        d = [random.choice(pos_lexicon) for j in range(length)]
        pos_docs.append(d)
    for j in range(n_neg):
        d = [random.choice(neg_lexicon) for j in range(length)]
        neg_docs.append(d)

    return pos_docs, neg_docs


# n_pos - number of positive doc_list
# n_neg - number of negative doc_list
# length - the length of each document.
def create_documents(n_pos, n_neg, length):
    plist, nlist = create_tokens(n_pos, n_neg, length)
    dlist = []
    for item in plist:
        d = Document(true_class="pos")
        d.add_tokens(item)
        dlist.append(d)
    for item in nlist:
        d = Document(true_class="neg")
        d.add_tokens(item)
        dlist.append(d)
    return dlist


def main():
    if not (os.path.exists("pos_lexicon.pkl") and os.path.exists("neg_lexicon.pkl")):
        neg_lexicon, pos_lexicon = generate_lexicons(100)
        save_lexicons("neg_lexicon.pkl", neg_lexicon, "pos_lexicon.pkl", pos_lexicon)


if __name__ == "__main__":
    main()
