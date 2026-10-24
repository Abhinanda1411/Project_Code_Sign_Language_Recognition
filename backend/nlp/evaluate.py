"""
evaluate.py - measures how much word prediction saves, on YOUR test sentences.

Usage (from the project root):
    python backend/nlp/evaluate.py [data/nlp/test_sentences.txt]

How it counts (for each word of each sentence):
  letter by letter : every letter has to be signed
  with prediction  : letters are signed until the word shows up in the suggestions,
                     then ONE action selects it. If it never shows up, every
                     letter is signed. (Space bar presses are the same either way
                     and are not counted.)
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from nlp.suggester import load_suggester, ROOT         
from nlp.build_data import tokens                       


def evaluate(suggester, sentences):
    letters = actions = hits = words_total = 0
    rows = []

    for sentence in sentences:
        words = tokens(sentence)
        typed = ""                                    # what is on screen so far
        s_letters = s_actions = 0

        for word in words:
            words_total += 1
            cost = len(word)                          # fallback: sign every letter
            used = False

            for i in range(len(word) + 1):
                suggestions = suggester.suggest(typed + word[:i])
                if word.upper() in suggestions:
                    if i + 1 < len(word):             # only worth it if it saves effort
                        cost = i + 1
                        used = True
                    break

            letters += len(word)
            actions += cost
            hits += used
            s_letters += len(word)
            s_actions += cost
            typed += word.upper() + " "

        rows.append((sentence.strip(), s_letters, s_actions))

    return letters, actions, hits, words_total, rows


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "nlp", "test_sentence.txt")
    if not os.path.exists(path):
        sys.exit(f"Test sentences file not found: {path}")

    suggester = load_suggester()
    if suggester is None:
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        sentences = [line for line in f if line.strip()]

    letters, actions, hits, words_total, rows = evaluate(suggester, sentences)
    saved = (1 - actions / letters) * 100 if letters else 0

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "nlp_eval.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "keystrokes"])
        w.writerow(["Letter by letter", letters])
        w.writerow(["With word prediction", actions])

    with open(os.path.join(ROOT, "results", "nlp_eval_details.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sentence", "letters_without", "actions_with"])
        w.writerows(rows)

    print(f"sentences: {len(sentences)} | words: {words_total}")
    print(f"letter by letter : {letters}")
    print(f"with prediction  : {actions}  ({saved:.1f}% fewer)")
    print(f"words completed from a suggestion: {hits} of {words_total}")


if __name__ == "__main__":
    main()
