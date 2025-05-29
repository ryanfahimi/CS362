import random
import sys

## ZeroR - our first learning algorithm.
### assume that list_of_examples is a list of strings. For example:
### ['outlook,temperature,humidity,windy,play\n', 'sunny,hot,high,FALSE,no\n', 'sunny,hot,high,TRUE,no\n', 'overcast,hot,high,FALSE,yes\n', 'rainy,mild,high,FALSE,yes\n', 'rainy,cool,normal,FALSE,yes\n', 'rainy,cool,normal,TRUE,no\n', 'overcast,cool,normal,TRUE,yes\n', 'sunny,mild,high,FALSE,no\n', 'sunny,cool,normal,FALSE,yes\n', 'rainy,mild,normal,FALSE,yes\n', 'sunny,mild,normal,TRUE,yes\n', 'overcast,mild,high,TRUE,yes\n', 'overcast,hot,normal,FALSE,yes\n', 'rainy,mild,high,TRUE,no\n']
### your code should get the last element in each string, which is the classification, and return the most common one.

def zeroR(list_of_examples) :
    classifications = [example.strip().split(",")[-1] for example in list_of_examples[1:]]
    return max(set(classifications), key=classifications.count) # you fix this.

### assume that list_of_examples is a list of strings. For example:
### ['outlook,temperature,humidity,windy,play\n', 'sunny,hot,high,FALSE,no\n', 'sunny,hot,high,TRUE,no\n', 'overcast,hot,high,FALSE,yes\n', 'rainy,mild,high,FALSE,yes\n', 'rainy,cool,normal,FALSE,yes\n', 'rainy,cool,normal,TRUE,no\n', 'overcast,cool,normal,TRUE,yes\n', 'sunny,mild,high,FALSE,no\n', 'sunny,cool,normal,FALSE,yes\n', 'rainy,mild,normal,FALSE,yes\n', 'sunny,mild,normal,TRUE,yes\n', 'overcast,mild,high,TRUE,yes\n', 'overcast,hot,normal,FALSE,yes\n', 'rainy,mild,high,TRUE,no\n']
### your code should get the last element in each string, which is the classification, and use random.choice() to select one and return it

def randR(list_of_examples) :
    classifications = [example.strip().split(",")[-1] for example in list_of_examples[1:]]
    return random.choice(classifications) # you fix this.



## Our main. We should be able to run from the command line like so:
## python ZeroR.py tennis.csv
## python ZeroR.py -z tennis.csv
## python ZeroR.py -r tennis.csv

if __name__ == '__main__' :

    classify_type = "-z"
    if len(sys.argv) < 2:
        print("Usage:  classify {-z|-r} file")
        sys.exit(-1)
    if len(sys.argv) == 3 :
        classify_type = sys.argv[1]
        if classify_type != "-z" and classify_type != "-r" :
            print("Usage:  classify {-z|-r} file")
            sys.exit(-1)
    f_name = sys.argv[-1]

    with open(f_name) as f :
        data = f.readlines()
        classifications = [line.strip().split(",")[-1] for line in data[1:]]
        if classify_type == "-z" :
            ## Uses ZeroR to find the most common classification, and then
            ## compares that value to the true classification for each line to compute accuracy.
            ## (fraction of answers that are correct.)
            most_common_classification = zeroR(data)
            accuracy = sum(1 for classification in classifications if classification == most_common_classification) / len(classifications)
            print(f"Accuracy: {accuracy}")
        else :
            ## For each line in the dataset, calls RandR to generate a prediction
            ## and compares that to the actual classification. Uses this to compute accuracy.
            ## (fraction of answers that are correct.)
            accuracy = sum(1 for classification in classifications if classification == randR(data)) / len(classifications)
            print(f"Accuracy: {accuracy}")
