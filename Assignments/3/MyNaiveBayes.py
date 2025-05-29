from collections import defaultdict
import pandas as pd


class MyNaiveBayes:

    # filename is the place where our data is stored.
    def __init__(self, filename, data=None, zero_r=False):
        self.filename = filename
        self.pos_examples = defaultdict()
        self.neg_examples = defaultdict()
        if data is None:
            self.data = pd.read_csv(
                self.filename,
                names=[
                    "Class",
                    "Age",
                    "Menopause",
                    "Tumor Size",
                    "Inv Nodes",
                    "Node Caps",
                    "Degree",
                    "Breast",
                    "Quadrant",
                    "Irradiated?",
                ],
            )
        else:
            self.data = data
        if zero_r:
            self.classify = self.zero_r
        else:
            self.classify = self.predict

    ## fit: fit the model to the examples.
    ## P(class | features) = alpha * P(features | class)P(class)
    ## We can compute P(class) by counting the fraction of classifications
    ## belonging to each class
    ## e.g. we want to store P('menopause'='premeno' | class='recurrence') in
    ## pos_examples, so we make a dict d = {'premeno':150, 'ge40':129, 'lt40':7}
    ## and set self.pos_examples['Menopause'] = d.
    ## (we'd actually set the values by iterating through the examples and counting
    ## occurrences for each label.
    ## this should use the data stored in self.data.
    def fit(self):
        ## for each attribute, we want to know the conditional probabilities of each feature given recurrence/no-recurrence
        ## In other words, what's P(tumor-size=0-4 | recurrence)?
        ## you can get those with the value_counts() method.
        ## We have two dictionaries in our object - one for positive examples (recurrence) and one for negative examples (no-recurrence)
        ## For each attribute, use the name (e.g. age) as the key and the value counts as the value for the dictionary.
        pos_rows = self.data[self.data["Class"] == "recurrence-events"]
        neg_rows = self.data[self.data["Class"] == "no-recurrence-events"]
        for attribute in self.data.columns[1:]:
            self.pos_examples[attribute] = dict(pos_rows[attribute].value_counts())
            self.neg_examples[attribute] = dict(neg_rows[attribute].value_counts())
        for attribute in self.data.columns[1:]:
            pos_sum = sum(self.pos_examples[attribute].values())
            neg_sum = sum(self.neg_examples[attribute].values())
            for key in self.pos_examples[attribute].keys():
                self.pos_examples[attribute][key] /= pos_sum
            for key in self.neg_examples[attribute].keys():
                self.neg_examples[attribute][key] /= neg_sum

    ## precict: examples can be a nested list of attributes, or else a Pandas dataframe.
    ## it should return a corresponding list of classifications.
    def predict(self, examples):
        ## To classify with Naive Bayes, do the following:
        ## for each example:
        ## P(recurrence | feature1, feature2, ...) = P(feature1, feature2, ... | recurrence)P(recurrence) / P(feature1,...)
        ## But, since we only care about which is more probable, we can drop the denominator.
        ## and then we apply the Naive Bayes assumption, so we need:
        ## P(feature1 | recurrence) * P(feature2 | recurrence) * ... * P(featuren | recurrence) * P(recurrence)
        ## P(feature1 | recurrence) is the value_count for feature1 found in fit, divided by the number of positive examples.
        ## (In other words, if we had 100 recurrences, and in those, 20 have tumor-size=0-4, P(recurrence | tumor-size=0-4)=20/100 = 0.2.
        ## P(recurrence) is the fraction of the set used to fit the data that are of that class.
        ## lastly, we need to use log-likelihood in order to avoid underflow, so we want to compute
        ## log(P(feature1 | recurrence)) + log(P(feature2 | recurrence)) + ... + log(P(featuren | recurrence)) + log(P(recurrence))
        ## we compute this for recurrence and no-recurrence, and select the larger one.
        classifications = []
        for index, example in examples.iterrows():
            pos_prob = len(self.data[self.data["Class"] == "recurrence-events"]) / len(
                self.data
            )
            neg_prob = len(
                self.data[self.data["Class"] == "no-recurrence-events"]
            ) / len(self.data)
            for attribute in self.data.columns[1:]:
                pos_prob *= self.pos_examples[attribute].get(example[attribute], 0)
                neg_prob *= self.neg_examples[attribute].get(example[attribute], 0)
            classifications.append(
                "recurrence-events" if pos_prob > neg_prob else "no-recurrence-events"
            )
        return classifications

    def zero_r(self, examples):
        return [self.data["Class"].value_counts().idxmax()] * len(examples)

    ## score: this method should take as input two lists or Series, one representing the predictions and one representing the true
    ## values, and return the F1 score. (2 * precision * recall) / (precision + recall) or
    ##                                   2 * true_positives / (2 * true_positive + false_positive + false_negative)
    def score(self, predicted_results, true_results):
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for predicted_result, true_result in zip(predicted_results, true_results):
            if (
                predicted_result == "recurrence-events"
                and true_result == "recurrence-events"
            ):
                true_positives += 1
            elif (
                predicted_result == "recurrence-events"
                and true_result == "no-recurrence-events"
            ):
                false_positives += 1
            elif (
                predicted_result == "no-recurrence-events"
                and true_result == "recurrence-events"
            ):
                false_negatives += 1
        return (
            2
            * true_positives
            / (2 * true_positives + false_positives + false_negatives)
        )

    ## five-fold
    def five_fold(self, zero_r=False):
        ## here is where you'll implement five-fold cross-validation. You should:
        ## 1. split the data into five equal "bins". (If the number is not divisible by 5, that's fine. Some bins can have one extra item.)
        ##  Note that you should not actually copy the data into new structures - just use indices to
        ##   keep track of which data is for training and which is for testing.
        ## 2. You'll do five iterations - for each iteration, 4 of the bins are training, and 1 is the test bin.
        ## In each iteration, create a new classifier and fit it to the training data.
        ##  3. Then test that classifier on the test data and compute F1.
        ## Once you're done, return the five F1 scores.
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        f1_scores = []
        k = 5
        fold_size = len(self.data) // k
        for i in range(k):
            test_start = i * fold_size
            if i == k - 1:
                test_end = len(self.data)
            else:
                test_end = (i + 1) * fold_size
            test_data = self.data[test_start:test_end]
            train_data = pd.concat([self.data[:test_start], self.data[test_end:]])
            classifier = MyNaiveBayes(self.filename, train_data, zero_r)
            classifier.fit()
            predictions = classifier.classify(test_data)
            f1_scores.append(classifier.score(predictions, test_data["Class"]))

        return f1_scores


if __name__ == "__main__":
    nb = MyNaiveBayes("breast-cancer.data")
    print(nb.five_fold())
    print(nb.five_fold(zero_r=True))
