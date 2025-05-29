import unittest
import tempfile
import os
import sys


from HMM import HMM, Sequence


class TestHMMMethods(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory to hold test files.
        self.temp_dir = tempfile.TemporaryDirectory()
        self.orig_dir = os.getcwd()
        os.chdir(self.temp_dir.name)

    def tearDown(self):
        os.chdir(self.orig_dir)
        self.temp_dir.cleanup()

    def create_model_files(self, basename, trans_content, emit_content):
        with open(f"{basename}.trans", "w") as f:
            f.write(trans_content)
        with open(f"{basename}.emit", "w") as f:
            f.write(emit_content)

    def test_load(self):
        trans_content = "# A 1.0\nA A 0.5\nA B 0.5\n"
        emit_content = "A x 0.7\nA y 0.3\nB x 0.4\nB y 0.6\n"
        basename = "load_test"
        self.create_model_files(basename, trans_content, emit_content)

        hmm = HMM()
        hmm.load(basename)

        self.assertIn("#", hmm.transitions)
        self.assertIn("A", hmm.transitions["#"])
        self.assertEqual(hmm.transitions["#"]["A"], 1.0)
        self.assertIn("A", hmm.transitions)
        self.assertAlmostEqual(hmm.transitions["A"]["A"], 0.5)
        self.assertAlmostEqual(hmm.transitions["A"]["B"], 0.5)

        self.assertIn("A", hmm.emissions)
        self.assertAlmostEqual(hmm.emissions["A"]["x"], 0.7)
        self.assertAlmostEqual(hmm.emissions["A"]["y"], 0.3)
        self.assertIn("B", hmm.emissions)
        self.assertAlmostEqual(hmm.emissions["B"]["x"], 0.4)
        self.assertAlmostEqual(hmm.emissions["B"]["y"], 0.6)

    def test_generate(self):
        # Set up a deterministic HMM: Only one possible transition and emission.
        transitions = {"#": {"A": 1.0}, "A": {"A": 1.0}}
        emissions = {"A": {"x": 1.0}}
        hmm = HMM(transitions=transitions, emissions=emissions)

        seq = hmm.generate(5)
        self.assertEqual(len(seq), 5)
        self.assertListEqual(seq.state_seq, ["A"] * 5)
        self.assertListEqual(seq.output_seq, ["x"] * 5)

    def test_forward(self):
        transitions = {"#": {"A": 1.0}, "A": {"A": 1.0}}
        emissions = {"A": {"x": 1.0}}
        hmm = HMM(transitions=transitions, emissions=emissions)

        seq = Sequence(state_seq=["A", "A", "A"], output_seq=["x", "x", "x"])
        final_state = hmm.forward(seq)
        self.assertEqual(final_state, "A")

    def test_viterbi(self):
        transitions = {"#": {"A": 1.0}, "A": {"A": 1.0}}
        emissions = {"A": {"x": 1.0}}
        hmm = HMM(transitions=transitions, emissions=emissions)

        seq = Sequence(state_seq=[], output_seq=["x", "x", "x"])
        tagged_seq = hmm.viterbi(seq)
        self.assertEqual(len(tagged_seq), 3)
        self.assertEqual(tagged_seq.state_seq, ["A", "A", "A"])
        self.assertEqual(tagged_seq.output_seq, seq.output_seq)

    def test_main_generate_only(self):
        basename = "test"
        trans_content = "# A 1.0\nA A 1.0\n"
        emit_content = "A x 1.0\n"
        self.create_model_files(basename, trans_content, emit_content)

        test_args = ["prog", basename]
        sys.argv = test_args

        hmm = HMM()
        hmm.main()
        obs_file = f"{basename}.obs"
        self.assertTrue(os.path.exists(obs_file))
        with open(obs_file, "r") as f:
            content = f.read().strip()
        outputs = content.split()
        self.assertEqual(len(outputs), 20)

    def test_main_generate_with_length(self):
        basename = "test"
        trans_content = "# A 1.0\nA A 1.0\n"
        emit_content = "A x 1.0\n"
        self.create_model_files(basename, trans_content, emit_content)

        test_args = ["prog", basename, "--generate", "10"]
        sys.argv = test_args

        hmm = HMM()
        hmm.main()
        obs_file = f"{basename}.obs"
        self.assertTrue(os.path.exists(obs_file))
        with open(obs_file, "r") as f:
            content = f.read().strip()
        outputs = content.split()
        self.assertEqual(len(outputs), 10)

    def test_main_with_observations_file(self):
        basename = "test"
        trans_content = "# A 1.0\nA A 1.0\n"
        emit_content = "A x 1.0\n"
        self.create_model_files(basename, trans_content, emit_content)

        obs_filename = "observations.txt"
        obs_content = "x x x"
        with open(obs_filename, "w") as f:
            f.write(obs_content)

        test_args = ["prog", basename, obs_filename]
        sys.argv = test_args

        hmm = HMM()
        hmm.main()
        with open(obs_filename, "r") as f:
            content = f.read().strip()
        self.assertEqual(content, obs_content)
        self.assertFalse(os.path.exists(f"{basename}.obs"))

    def test_main_viterbi(self):
        basename = "test"
        trans_content = "# A 1.0\nA A 1.0\n"
        emit_content = "A x 1.0\n"
        self.create_model_files(basename, trans_content, emit_content)

        test_args = ["prog", basename, "--viterbi"]
        sys.argv = test_args

        hmm = HMM()
        hmm.main()

        tagged_obs_file = f"{basename}.tagged.obs"
        self.assertTrue(os.path.exists(tagged_obs_file))
        with open(tagged_obs_file, "r") as f:
            lines = f.read().strip().splitlines()
        self.assertEqual(len(lines), 2)
        state_seq = lines[0].split()
        obs_seq = lines[1].split()
        self.assertEqual(len(state_seq), 20)
        self.assertEqual(len(obs_seq), 20)


if __name__ == "__main__":
    unittest.main()

