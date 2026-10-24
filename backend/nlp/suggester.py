"""
suggester.py - word prediction for fingerspelled text.

Three things, all driven by plain word counts (no training, no black box):

  1. WORD COMPLETION   typed "HEL"           -> HELLO, HELP, ...        (most common first)
  2. NEXT WORD         typed "HOW ARE "      -> YOU, THEY, ...          (from word-pair counts)
  3. LETTER TOLERANCE  typed "QLEASE" (P read as Q) -> PLEASE, ...      (letters that the
                       recogniser confuses are allowed to be swapped for one letter)

Data (made by build_data.py from an openly licensed sentence collection):
  data/nlp/unigrams.tsv   word <TAB> count
  data/nlp/bigrams.tsv    word1 word2 <TAB> count      ("<s>" = start of a sentence)
"""
import heapq
import math
import os
from bisect import bisect_left
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DIR = os.path.join(ROOT, "data", "nlp")

START = "<s>"

# Letters the recogniser mixes up (start: pairs that look alike in ASL, and the
# pairs seen in the earlier prototype's confusion matrix). Each pair works both ways.
# Tune this from YOUR validation errors, not from the test session.
CONFUSABLE_PAIRS = [("P", "Q"), ("P", "K"), ("M", "N"), ("M", "T"), ("N", "T"),
                    ("U", "R"), ("U", "V"), ("R", "V"), ("A", "S"), ("E", "S"),
                    ("D", "X"), ("I", "Y")]

BIGRAM_WEIGHT = 1.0        # how much "this word often follows the previous one" counts
FUZZY_PENALTY = 3.0        # a swapped-letter match ranks below a normal match
CANDIDATES = 40            # words looked at per prefix before re-ranking


def _confusable_map(pairs):
    out = defaultdict(set)
    for a, b in pairs:
        out[a.lower()].add(b.lower())
        out[b.lower()].add(a.lower())
    return {k: sorted(v) for k, v in out.items()}


class Suggester:
    def __init__(self, unigrams, bigrams, pairs=None, k=5):
        self.k = k
        self.uni = dict(unigrams)
        self.words = sorted(self.uni)                     # alphabetical, for prefix search
        self.confusable = _confusable_map(CONFUSABLE_PAIRS if pairs is None else pairs)

        self.big = dict(bigrams)
        follow = defaultdict(list)
        for (w1, w2), n in self.big.items():
            follow[w1].append((n, w2))
        self.follow = {w1: [w for _, w in heapq.nlargest(20, lst)] for w1, lst in follow.items()}
        self.top_words = [w for _, w in heapq.nlargest(20, ((n, w) for w, n in self.uni.items()))]

    # ---------------- helpers ----------------
    def _with_prefix(self, prefix):
        lo = bisect_left(self.words, prefix)
        hi = bisect_left(self.words, prefix + "￿")
        span = self.words[lo:hi]
        return heapq.nlargest(CANDIDATES, span, key=self.uni.get)

    def _score(self, word, prev):
        s = math.log(self.uni[word])
        n = self.big.get((prev, word), 0)
        if n:
            s += BIGRAM_WEIGHT * math.log1p(n)
        return s

    # ---------------- main entry ----------------
    def suggest(self, text):
        """text: what has been typed so far (any case). Returns up to k words, UPPERCASE."""
        words = text.lower().split(" ")
        partial = words[-1]
        done = [w for w in words[:-1] if w]
        prev = done[-1] if done else START

        # nothing typed in the current word -> predict the NEXT word
        if partial == "":
            ranked = self.follow.get(prev) or self.follow.get(START) or self.top_words
            return [w.upper() for w in ranked[: self.k]]

        scored = {}
        for w in self._with_prefix(partial):
            scored[w] = self._score(w, prev)

        # too few matches -> also try swapping ONE typed letter for a look-alike letter
        if len(scored) < self.k:
            for i, ch in enumerate(partial):
                for alt in self.confusable.get(ch, ()):
                    for w in self._with_prefix(partial[:i] + alt + partial[i + 1:]):
                        s = self._score(w, prev) - FUZZY_PENALTY
                        if w not in scored or s > scored[w]:
                            scored[w] = s

        best = heapq.nlargest(self.k, scored.items(), key=lambda kv: kv[1])
        return [w.upper() for w, _ in best]


def _read_tsv(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            key, _, num = line.rstrip("\n").rpartition("\t")
            if key and num.isdigit():
                yield key, int(num)


def load_suggester(directory=DEFAULT_DIR, k=5):
    """Suggester from data/nlp/, or None (with a message) if the files are not there yet."""
    uni_path = os.path.join(directory, "unigrams.tsv")
    big_path = os.path.join(directory, "bigrams.tsv")

    if not (os.path.exists(uni_path) and os.path.exists(big_path)):
        print("Word prediction is OFF: run  python backend/nlp/build_data.py <sentences file>  first "
              f"(expected {uni_path} and {big_path}).")
        return None

    unigrams = dict(_read_tsv(uni_path))
    bigrams = {tuple(key.split(" ")): n for key, n in _read_tsv(big_path)}
    return Suggester(unigrams, bigrams, k=k)