import random

## for each element, the first six items are the input, and the last is the
## expected output.

training_examples = [
    [1, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1],
    [1, 1, 0, 0, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0, 1, 1],
    [0, 1, 1, 0, 1, 1, 0],
    [0, 0, 0, 1, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 1, 0],
    [0, 1, 1, 1, 0, 0, 0],
]

def threshold(val) :
    if val < 0 :
        return 0
    else :
        return 1

def perceptron_training() :
    alpha = 0.1
    bias = -0.1
    weights = [0] * 6
    ## each weight will be between -0.05 and 0.05
    for i in range(5) :
        weights[i] = (random.random() / 10) - 0.05

    converged = False
    while not converged :
        converged = True
        for example in training_examples :
            ## you complete this part.
            ## first, compute actual output
            inputs = example[:-1]
            expected_output = example[-1]
            total = bias
            for i in range(len(inputs)):
                total += weights[i] * inputs[i]

            actual_output = threshold(total)

            ## next, update weights
            error = expected_output - actual_output
            if error != 0:
                converged = False
                for i in range(len(weights)):
                    weights[i] += alpha * inputs[i] * error

    ## print results
    for example in training_examples :
        expected_output = example[-1]
        inputs = example[:-1]
        total = bias
        for i in range(len(inputs)):
            total += weights[i] * inputs[i]
        actual_output = threshold(total)
        print(f"Expected: {expected_output} Actual: {actual_output}")

if __name__ == "__main__":
    perceptron_training()