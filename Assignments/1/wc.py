import argparse
import os
import pickle
import re
from string import punctuation

def wordfreq(f_name, strip_punc=False, to_lower=False, remove_non_words=False, separators=None) :
    word_dict = {}
    try:
        with open(f_name) as f:
            text = f.read()

            if separators:
                pattern = '[' + re.escape(separators) + ']+'
                words = re.split(pattern, text)
            else:
                words = text.split()

            for word in words:
                if strip_punc:
                    word = word.strip(punctuation)
                if to_lower:
                    word = word.lower()

                if remove_non_words:
                    if not word.isalpha():
                        continue

                # Skip empty words
                if not word:
                    continue

                if word in word_dict:
                    word_dict[word] += 1
                else:
                    word_dict[word] = 1

    except (UnicodeDecodeError, PermissionError, IsADirectoryError):
        print(f"Unable to open file: {f_name}")

    return word_dict

def combine_dicts(master_dict, new_dict):
    for word, count in new_dict.items():
        master_dict[word] = master_dict.get(word, 0) + count

def main():
    parser = argparse.ArgumentParser(description="Word frequency counter.")
    parser.add_argument(
        "path",
        help="File or directory to process (if dir, process all text files)."
    )
    parser.add_argument(
        "--strip",
        action="store_true",
        help="Strip punctuation from the start/end of words."
    )
    parser.add_argument(
        "--lower",
        action="store_true",
        help="Make all words lowercase."
    )
    parser.add_argument(
        "--nonwords",
        action="store_true",
        help="Remove words if they have any non-alphabetic chars."
    )
    parser.add_argument(
        "--separator",
        type=str,
        default=None,
        help="Use these characters as delimiters; if not given, splits on whitespace."
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save word frequencies to a file (instead of printing)."
    )

    args = parser.parse_args()

    total_freq = {}

    if os.path.isfile(args.path):
        total_freq = wordfreq(
            args.path,
            strip_punc=args.strip,
            to_lower=args.lower,
            remove_non_words=args.nonwords,
            separators=args.separator
        )

    else:
        for root, dirs, files in os.walk(args.path):
            for filename in files:
                fullpath = os.path.join(root, filename)
                freq = wordfreq(
                    fullpath,
                    strip_punc=args.strip,
                    to_lower=args.lower,
                    remove_non_words=args.nonwords,
                    separators=args.separator
                )
                combine_dicts(total_freq, freq)

    if args.save:
        with open(args.save, 'wb') as pf:
            pickle.dump(total_freq, pf)
    else:
        print(total_freq)


if __name__ == "__main__":
    main()




