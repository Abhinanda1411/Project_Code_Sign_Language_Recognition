"""
build_data.py - makes the word-prediction data from a sentence collection.

Usage (from the project root):
    python backend/nlp/build_data.py data/nlp/eng_sentences [--holdout data/nlp/test_sentences.txt]

PATH_TO_SENTENCES: the Tatoeba English sentences file (tab-separated:
id <TAB> language <TAB> sentence), as .tsv, .bz2, .gz, .tar.bz2, .zip - or any
plain text file with one sentence per line.

--holdout: sentences you will TEST with. Any identical sentence in the collection
is left out of the counts, so the test is fair (the model has never seen it).

Writes data/nlp/unigrams.tsv and data/nlp/bigrams.tsv.
Only the letters a-z are kept (that is all fingerspelling can produce):
punctuation and digits split words, apostrophes are removed (don't -> dont).
"""
import argparse
import bz2
import gzip
import io
import os
import re
import sys
import tarfile
import zipfile
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT_DIR = os.path.join(ROOT, "data", "nlp")

MAX_WORDS = 60000           # vocabulary size
MIN_WORD_COUNT = 2
MIN_BIGRAM_COUNT = 2
START = "<s>"


def open_text(path):
    """Open .tsv / .bz2 / .gz / .tar.* / .zip / plain text as a text stream."""
    with open(path, "rb") as f:
        head = f.read(4)

    if zipfile.is_zipfile(path):
        z = zipfile.ZipFile(path)
        name = [n for n in z.namelist() if not n.endswith("/")][0]
        return io.TextIOWrapper(z.open(name), encoding="utf-8", errors="replace")

    if tarfile.is_tarfile(path):
        t = tarfile.open(path)
        member = [m for m in t.getmembers() if m.isfile()][0]
        return io.TextIOWrapper(t.extractfile(member), encoding="utf-8", errors="replace")

    if head[:3] == b"BZh":
        return io.TextIOWrapper(bz2.open(path, "rb"), encoding="utf-8", errors="replace")
    if head[:2] == b"\x1f\x8b":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")

    return open(path, encoding="utf-8", errors="replace")


def sentence_of(line):
    """Tatoeba lines are id<TAB>lang<TAB>text; plain files are just the text."""
    parts = line.rstrip("\n").split("\t")
    if len(parts) >= 3:
        if parts[1] not in ("eng", "en"):
            return None
        return parts[2]
    return parts[0]


def tokens(sentence):
    s = sentence.lower().replace("'", "").replace("’", "")
    return re.findall(r"[a-z]+", s)


def key_of(sentence):
    return " ".join(tokens(sentence))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sentences")
    ap.add_argument("--holdout", default=None)
    args = ap.parse_args()

    held = set()
    if args.holdout:
        with open(args.holdout, encoding="utf-8") as f:
            held = {key_of(line) for line in f if line.strip()}
        print(f"holding out {len(held)} test sentences")

    uni, big = Counter(), Counter()
    n_sent = skipped = 0

    with open_text(args.sentences) as f:
        for line in f:
            s = sentence_of(line)
            if not s:
                continue
            words = tokens(s)
            if not words:
                continue
            if " ".join(words) in held:
                skipped += 1
                continue
            n_sent += 1
            uni.update(words)
            prev = START
            for w in words:
                big[(prev, w)] += 1
                prev = w

    vocab = {w for w, n in uni.most_common(MAX_WORDS) if n >= MIN_WORD_COUNT}
    if not vocab:
        sys.exit("No words found - is this the right file?")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "unigrams.tsv"), "w", encoding="utf-8") as f:
        for w, n in uni.most_common():
            if w in vocab:
                f.write(f"{w}\t{n}\n")

    kept = 0
    with open(os.path.join(OUT_DIR, "bigrams.tsv"), "w", encoding="utf-8") as f:
        for (a, b), n in big.most_common():
            if n >= MIN_BIGRAM_COUNT and b in vocab and (a == START or a in vocab):
                f.write(f"{a} {b}\t{n}\n")
                kept += 1

    print(f"sentences used: {n_sent}  (left out as test sentences: {skipped})")
    print(f"vocabulary: {len(vocab)} words | word pairs kept: {kept}")
    print("wrote", os.path.relpath(OUT_DIR, ROOT))


if __name__ == "__main__":
    main()