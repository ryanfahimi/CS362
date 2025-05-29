import random
import argparse
import os
import numpy
from black.trans import defaultdict


# Sequence - represents a sequence of hidden states and corresponding
# output variables.

class Sequence:
    def __init__(self, state_seq, output_seq):
        self.state_seq  = state_seq   # sequence of states
        self.output_seq = output_seq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.state_seq)+ '\n'+ ' '.join(self.output_seq)+ '\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.output_seq)

# HMM model
class HMM:
    def __init__(self, transitions=None, emissions=None):
        """creates a model from transition and emission probabilities
        e.g. {'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'},
              'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'},
              'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}}"""
        self.transitions = transitions if transitions is not None else {}
        self.emissions = emissions if emissions is not None else {}

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""
        with open(f"{basename}.trans", "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                from_state, to_state, prob = line.split()
                if from_state not in self.transitions:
                    self.transitions[from_state] = defaultdict(float)
                self.transitions[from_state][to_state] = float(prob)


        with open(f"{basename}.emit", "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                state, output, prob = line.split()
                if state not in self.emissions:
                    self.emissions[state] = defaultdict(float)
                self.emissions[state][output] = float(prob)

   ## you do this.
    def generate(self, n):
        """return an n-length Sequence by randomly sampling from this HMM."""
        current_state = '#'
        state_seq = []
        output_seq = []
        for _ in range(n):
            next_states = list(self.transitions[current_state].keys())
            transition_probs = list(self.transitions[current_state].values())
            current_state = random.choices(next_states, transition_probs)[0]
            state_seq.append(current_state)

            outputs = list(self.emissions[current_state].keys())
            emission_probs = list(self.emissions[current_state].values())
            output = random.choices(outputs, emission_probs)[0]
            output_seq.append(output)
        return Sequence(state_seq, output_seq)

    def forward(self, sequence):
    ## you do this: Implement the forward algorithm. Given a Sequence with a list of emissions,
    ## determine the most likely final state.
        matrix = [[0.0 for _ in range(len(self.emissions))] for _ in range(len(sequence))]
        for i, state in enumerate(self.emissions):
                matrix[0][i] = self.transitions['#'][state] * self.emissions[state][sequence.output_seq[0]]

        for i in range(1, len(sequence)):
            for j, state in enumerate(self.emissions):
                for k, prev_state in enumerate(self.emissions):
                    matrix[i][j] += matrix[i - 1][k] * self.transitions[prev_state][state] * self.emissions[state][sequence.output_seq[i]]

        states = list(self.emissions.keys())
        return states[numpy.argmax(matrix[-1])]


    def viterbi(self, sequence):
        ## You do this. Given a sequence with a list of emissions, fill in the most likely
        ## hidden states using the Viterbi algorithm.
        #max instead of sum
        matrix = [[0.0 for _ in range(len(self.emissions))] for _ in range(len(sequence))]
        back_pointers = [[0 for _ in range(len(self.emissions))] for _ in range(len(sequence))]
        for i, state in enumerate(self.emissions):
                matrix[0][i] = self.transitions['#'][state] * self.emissions[state][sequence.output_seq[0]]

        for i in range(1, len(sequence)):
            for j, state in enumerate(self.emissions):
                max_prob = 0
                max_state = 0
                for k, prev_state in enumerate(self.emissions):
                    prob = matrix[i - 1][k] * self.transitions[prev_state][state] * self.emissions[state][sequence.output_seq[i]]
                    if prob > max_prob:
                        max_prob = prob
                        max_state = k
                matrix[i][j] = max_prob
                back_pointers[i][j] = max_state

        best_sequence = []
        best_state = numpy.argmax(matrix[-1])
        states = list(self.emissions.keys())
        best_sequence.append(states[best_state])
        for i in range(len(sequence) - 1, 0, -1):
            best_state = back_pointers[i][best_state]
            best_sequence.append(states[best_state])
        best_sequence.reverse()

        return Sequence(best_sequence, sequence.output_seq)



    def main(self):
        parser = argparse.ArgumentParser(description="Hidden Markov Model.")
        parser.add_argument(
            "basename",
            help="Base name of the model files",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--generate",
            help="Generate a sequence of length n",
            type=int,
            default=20,
        )
        parser.add_argument(
            "--forward",
            action="store_true",
            help="Use the forward algorithm",
        )
        parser.add_argument(
            "--viterbi",
            action="store_true",
            help="Use the Viterbi algorithm",
        )
        parser.add_argument(
            "observations",
            help="observations",
            type=str,
            nargs="?",
        )
        args = parser.parse_args()
        if args.basename is None:
            print("Please provide a base name for the model files.")
            return
        self.load(args.basename)
        if args.observations is not None:
            if os.path.exists(args.observations):
                with open(args.observations, "r") as f:
                    sequence = Sequence(state_seq=[], output_seq=f.read().split())
            else:
                sequence = self.generate(args.generate)
                with open(args.observations, "w") as f:
                    f.write(" ".join(sequence.output_seq))
        else:
            sequence = self.generate(args.generate)
            with open(f"{args.basename}.obs", "w") as f:
                f.write(" ".join(sequence.output_seq))

        if args.forward:
            final_state = self.forward(sequence)
            print(f"Most likely final state: {final_state}")
        if args.viterbi:
            best_sequence = self.viterbi(sequence)
            with open(f"{args.basename}.tagged.obs", "w") as f:
                f.write(f"{' '.join(best_sequence.state_seq)}\n")
                f.write(f"{' '.join(sequence.output_seq)}")



if __name__ == "__main__":
    hmm = HMM()
    hmm.main()






