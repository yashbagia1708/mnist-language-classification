# MNIST and Language Classification

[![ci](https://github.com/yashbagia1708/mnist-language-classification/actions/workflows/ci.yml/badge.svg)](https://github.com/yashbagia1708/mnist-language-classification/actions/workflows/ci.yml)

Logistic regression on two very different inputs:

1. **MNIST, "is this digit a 6?"** on all 784 pixels
2. **The same task after PCA**, keeping 88% of the variance
3. **A tiny language classifier** (English, French, Spanish, German) on
   TF-IDF word features

Built for a university machine learning unit.

## Results

MNIST: 60,000 training and 10,000 test images; 958 of the test images are 6s.

| Model | Features | Test errors | Recall on 6s |
|---|---|---|---|
| Logistic regression | 784 pixels | 140 / 10,000 | 0.92 |
| Logistic regression after PCA | 74 components (88% of variance) | 143 / 10,000 | 0.91 |

**PCA cuts the features by 90% for almost no loss.** 74 components do the
job of 784 pixels, with 3 more mistakes in 10,000.

**"99% accuracy" says less than it sounds.** Only 9.6% of the test images
are 6s, so a model that always answered "not a 6" would score 90%. The
useful numbers are the errors on the 6s themselves: 79 of 958 real 6s were
missed (the confusion matrix and the per-digit error counts are printed by
the script).

**The language classifier doesn't work, and the reason is the data.** It
has 12 sentences, 3 per language. With 9 for training and 3 for testing, the
test sentences share no words with the training ones, so none of their words
carries any learned weight and the model falls back to the same class
(German) for all three: 0 of 3 correct. Character n-grams don't rescue it either (at most 25% in
leave-one-out). It's kept as a demonstration of the TF-IDF pipeline; a real
version needs a real corpus, hundreds of sentences per language.

## Running it

```bash
pip install -r requirements.txt
python classification.py          # downloads MNIST once (11 MB) into data/
python tests/check_results.py     # same run, with checks on the headline results (CI runs this)
```

The full run takes a few minutes; most of it is fitting logistic regression
on 60,000 images.

## Changes since the original submission

Fixed after the original upload (see the pull requests for detail):

- **PCA was fitted on the test set separately**, so test images were
  projected onto different components than the model was trained on. It's
  now fitted on the training set and applied to both.
- The PCA part re-downloaded MNIST from OpenML with a different split (and
  that download now fails with current pandas). It uses the same split as
  part 1, so the two rows in the table compare directly.
- MNIST is read from the Keras `mnist.npz` file directly, so TensorFlow is no
  longer a dependency.
- Several printed results were mislabelled: misclassified "digits" showed
  0/1 labels, the language part's misclassified labels came from the wrong
  list, and its "principal components" line was really the vocabulary size.
